from flask import Flask, render_template, request, jsonify, send_from_directory, redirect, url_for, session, flash
import pandas as pd
import numpy as np
from pathlib import Path
from werkzeug.utils import secure_filename
from PIL import Image
import base64, io, traceback, uuid, csv, os
from PIL import ImageOps, UnidentifiedImageError

app = Flask(__name__)
# Demo login configuration. Override FLASK_SECRET_KEY in real deployments.
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "demo-only-change-this-secret-key")
USERS_CSV = Path(__file__).resolve().parent / "users.csv"

@app.before_request
def require_login():
    # Allow login/logout and Flask's static assets without an authenticated session.
    if request.endpoint in ("login", "static"):
        return None
    if not session.get("user_id"):
        return redirect(url_for("login", next=request.path))

@app.route("/login", methods=["GET", "POST"])
def login():
    # If the user already has a valid session, open AI extraction directly.
    if session.get("user_id"):
        return redirect(url_for("ai_extraction"))
    error = None
    if request.method == "POST":
        uid = request.form.get("uid", "").strip()
        password = request.form.get("password", "")
        valid = False
        try:
            with USERS_CSV.open("r", newline="", encoding="utf-8-sig") as f:
                for row in csv.DictReader(f):
                    if row.get("uid", "").strip() == uid and row.get("password", "") == password:
                        valid = True
                        break
        except FileNotFoundError:
            error = "Login configuration is missing. Please contact the administrator."
        if valid:
            session.clear()
            session["user_id"] = uid
            # After successful authentication, go directly to AI extraction.
            return redirect(url_for("ai_extraction"))
        if error is None:
            error = "Invalid user ID or password."
    return render_template("login.html", error=error)

@app.route("/logout", methods=["GET", "POST"])
def logout():
    session.clear()
    return redirect(url_for("login"))
APP_VERSION = "V8"
# Reject oversized uploads early to avoid exhausting memory during image inference.
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

REQUIRED_COLUMNS = {
    "parcel_id", "building_id", "floor_no", "unit_id", "property_type",
    "area_sq_m", "latitude", "longitude", "base_elevation_m",
    "floor_height_m", "unit_elevation_m", "building_length_m",
    "building_width_m", "use_type", "owner_id"
}

# V8: cache the trained PyTorch U-Net in memory after its first use.
SEGMENTOR = None
PROCESSOR = None
AI_MODEL_ERROR = None
MODEL_PATH = BASE_DIR / "models" / "unet_resnet34_whu_best.pth"
MODEL_INPUT_SIZE = 512
MODEL_THRESHOLD = 0.50
MODEL_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
MODEL_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def prototype_ulpin(row):
    return f"AP-KR-{row['parcel_id']}-{row['building_id']}-F{int(row['floor_no']):02d}-U{row['unit_id']}"

def load_and_prepare(path):
    df = pd.read_csv(path)
    missing = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))

    numeric_cols = [
        "floor_no", "area_sq_m", "latitude", "longitude",
        "base_elevation_m", "floor_height_m", "unit_elevation_m",
        "building_length_m", "building_width_m"
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if df[numeric_cols].isna().any().any():
        raise ValueError("Some required numeric fields contain invalid or missing values.")

    df["prototype_ulpin"] = df.apply(prototype_ulpin, axis=1)

    building_ids = list(df["building_id"].astype(str).drop_duplicates())
    building_offsets = {bid: i * 52.0 for i, bid in enumerate(building_ids)}

    records = []
    for _, r in df.iterrows():
        unit_text = str(r["unit_id"])
        letter = unit_text[-1].upper() if unit_text else "A"
        unit_index = max(0, min(3, ord(letter) - ord("A")))
        half_w = float(r["building_width_m"]) / 2
        half_l = float(r["building_length_m"]) / 2
        col = unit_index % 2
        row2 = unit_index // 2
        unit_w = half_w
        unit_l = half_l
        x0 = building_offsets[str(r["building_id"])] + col * unit_w
        y0 = row2 * unit_l
        z0 = (int(r["floor_no"]) - 1) * float(r["floor_height_m"])

        rec = r.to_dict()
        rec.update({
            "x": round(x0, 3), "y": round(y0, 3), "z": round(z0, 3),
            "dx": round(unit_w, 3), "dy": round(unit_l, 3),
            "dz": round(float(r["floor_height_m"]), 3),
            "building_offset_x": building_offsets[str(r["building_id"])]
        })
        records.append(rec)
    return pd.DataFrame(records)

def make_payload(df):
    parcel_map = {}
    for bid, group in df.groupby("building_id", sort=False):
        r = group.iloc[0]
        x = float(r["building_offset_x"])
        y = 0.0
        length = float(r["building_length_m"])
        width = float(r["building_width_m"])
        parcel_map[str(r["parcel_id"])] = {
            "parcel_id": str(r["parcel_id"]), "building_id": str(bid),
            "x": x - 2, "y": y - 2, "length": length + 4, "width": width + 4
        }
    return {
        "summary": {
            "records": int(len(df)),
            "parcels": int(df["parcel_id"].nunique()),
            "buildings": int(df["building_id"].nunique()),
            "max_floors": int(df.groupby("building_id")["floor_no"].max().max())
        },
        "units": df.to_dict(orient="records"),
        "parcels_geometry": list(parcel_map.values())
    }

@app.route("/version")
def version():
    return jsonify({"version": APP_VERSION, "app": "3D ULPIN Flask", "purpose": "Prototype AI footprint extraction and 3D property mapping"})

@app.route("/")
def home():
    # Opening the application root should take authenticated users directly
    # to AI Extraction. Unauthenticated users are redirected to login by
    # require_login() above.
    return redirect(url_for("ai_extraction"))

@app.route("/dashboard")
def dashboard():
    # Friendly dashboard URL; keep the existing dashboard template unchanged.
    return render_template("index.html")

@app.route("/new-project")
def new_project():
    # Separate five-stage project workflow UI.
    return render_template("new_project.html")

@app.route("/ai-extraction")
def ai_extraction():
    return render_template("ai_extraction.html")


@app.route("/gis-3d")
def gis_3d():
    return render_template("gis_3d.html")

@app.route("/sample_aerial.png")
def sample_aerial():
    return send_from_directory(BASE_DIR, "sample_aerial.png")

@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_DIR, filename)

@app.route("/build-3d", methods=["POST"])
def build_3d():
    try:
        payload = request.get_json(force=True)
        polygons = payload.get("polygons", [])
        image_width = float(payload.get("image_width", 1536))
        image_height = float(payload.get("image_height", 1024))
        if not polygons:
            return jsonify({"error": "No building polygons were supplied. Run AI extraction first."}), 400

        # V7: preserve the actual AI footprint geometry and convert pixel coordinates
        # into a local metric coordinate system. In production, scale/origin should
        # come from a georeferenced orthophoto, GNSS/GCP or official GIS layer.
        scale_m_per_px = float(payload.get("scale_m_per_px", 0.08))
        parcel_margin_m = float(payload.get("parcel_margin_m", 0.0))
        buildings, floors = [], []

        def polygon_area(points):
            return abs(sum(points[i][0]*points[(i+1)%len(points)][1] - points[(i+1)%len(points)][0]*points[i][1] for i in range(len(points))) / 2.0)

        for i, poly in enumerate(polygons, 1):
            pts = poly.get("polygon_pixels", [])
            if len(pts) < 3:
                continue
            xs=[float(q[0]) for q in pts]; ys=[float(q[1]) for q in pts]
            minx,maxx,miny,maxy=min(xs),max(xs),min(ys),max(ys)

            # Keep the exact detected polygon shape in local metres.
            local_poly = [
                [round((float(px)-minx)*scale_m_per_px, 3),
                 round((maxy-float(py))*scale_m_per_px, 3)]
                for px, py in pts
            ]
            footprint_area = max(polygon_area(local_poly), 0.5)
            width = max(0.1, max(q[0] for q in local_poly)-min(q[0] for q in local_poly))
            depth = max(0.1, max(q[1] for q in local_poly)-min(q[1] for q in local_poly))

            # Prototype height only. Replace with LiDAR/DSM/floor-plan height later.
            floor_count=max(1,min(5,int(round(2 + min(footprint_area/140.0, 3)))))
            floor_h=3.0
            height=floor_count*floor_h
            parcel_id=f"P{i:03d}"; building_id=f"B{i:03d}"
            origin_x=minx*scale_m_per_px
            origin_y=(image_height-maxy)*scale_m_per_px

            # V8.1 boundary correction: do not expand the parcel outline beyond
            # the detected building footprint. The prior radial 1.5 m expansion
            # made the outline visibly larger than the roof and was not a surveyed
            # parcel boundary. Keep the prototype outline coincident with footprint.
            # parcel_margin_m is retained for API compatibility but defaults to 0.
            if parcel_margin_m > 0:
                # Explicitly ignore outward expansion for footprint-aligned display.
                parcel_margin_m = 0.0
            parcel_poly = [q[:] for q in local_poly]

            center_lat=16.50 + (image_height/2 - (miny+maxy)/2)*scale_m_per_px/111000
            center_lon=80.65 + ((minx+maxx)/2-image_width/2)*scale_m_per_px/(111000*0.96)
            building={
                "building_id":building_id,"parcel_id":parcel_id,
                "area_sq_m":round(footprint_area,2),
                "footprint_width_m":round(width,2),"footprint_depth_m":round(depth,2),
                "height_m":height,"floors":floor_count,
                "center_lat":round(center_lat,6),"center_lon":round(center_lon,6),
                "polygon_pixels":pts,"local_polygon":local_poly,
                "x":round(origin_x,2),"y":round(origin_y,2),
                "parcel_polygon":parcel_poly,
                "geometry_source":"Actual AI footprint polygon"
            }
            buildings.append(building)

            # Four prototype units per floor for demonstrating vertical ownership.
            # Unit geometry is intentionally not claimed as cadastral geometry yet.
            for f in range(1,floor_count+1):
                for uidx in range(4):
                    unit_id=f"{f}{chr(65+uidx)}"
                    floors.append({
                        "parcel_id":parcel_id,"building_id":building_id,"floor_no":f,"unit_id":unit_id,
                        "unit_area_sq_m":round(max(4.0,footprint_area/4.0),2),
                        "elevation_m":round((f-1)*floor_h,2),
                        "prototype_3d_ulpin":f"3D-P-{parcel_id}-{building_id}-F{f:02d}-U{uidx+1:02d}"
                    })

        return jsonify({
            "success":True,"scale_m_per_px":scale_m_per_px,
            "buildings":buildings,"floors":floors,
            "summary":{"buildings":len(buildings),"parcels":len(buildings),"units":len(floors),
                       "max_floors":max([b["floors"] for b in buildings],default=0)},
            "note":"V8.1 preserves the supplied AI footprint polygon and extrudes that geometry into a 3D building shell. The displayed prototype outline is coincident with the detected footprint and is not an official parcel boundary. Georeferencing, official parcel boundaries, topology validation and measured building height remain production GIS tasks."
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error":str(e)}),500

@app.route("/sample")
def sample():
    try:
        return jsonify(make_payload(load_and_prepare(BASE_DIR / "sample_data.csv")))
    except Exception as e:
        return jsonify({"error": str(e)}), 400

def load_ai_model():
    """Load the trained SMP U-Net/ResNet34 checkpoint once; CPU is always supported."""
    global SEGMENTOR, PROCESSOR, AI_MODEL_ERROR
    if SEGMENTOR is not None:
        return SEGMENTOR, PROCESSOR
    if AI_MODEL_ERROR:
        raise RuntimeError(AI_MODEL_ERROR)
    try:
        import torch
        import segmentation_models_pytorch as smp

        if not MODEL_PATH.is_file():
            raise FileNotFoundError(
                f"Trained checkpoint is missing: {MODEL_PATH}. "
                "Place unet_resnet34_whu_best.pth in the application's models folder."
            )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights=None,
            in_channels=3,
            classes=1,
            activation=None,
        )
        checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
        state = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        model.load_state_dict(state, strict=True)
        model.to(device)
        model.eval()
        SEGMENTOR = model
        PROCESSOR = {"device": device, "input_size": MODEL_INPUT_SIZE,
                     "checkpoint": str(MODEL_PATH),
                     "checkpoint_epoch": checkpoint.get("epoch") if isinstance(checkpoint, dict) else None}
        return SEGMENTOR, PROCESSOR
    except Exception as e:
        AI_MODEL_ERROR = (
            "The trained U-Net checkpoint could not be loaded. Verify that PyTorch and "
            "segmentation-models-pytorch are installed and that the checkpoint matches "
            "the U-Net/ResNet34 architecture. Technical detail: " + str(e)
        )
        raise RuntimeError(AI_MODEL_ERROR)


def _onnx_input_size(_model):
    """V8 trained checkpoint inference uses fixed 512 x 512 tiles."""
    return MODEL_INPUT_SIZE, MODEL_INPUT_SIZE


def _run_building_tile(model, tile_rgb, input_h, input_w):
    """Resize RGB tile, apply ImageNet normalization, and return sigmoid probabilities."""
    import cv2
    import torch

    resized = cv2.resize(tile_rgb, (input_w, input_h), interpolation=cv2.INTER_LINEAR)
    arr = resized.astype(np.float32) / 255.0
    arr = (arr - MODEL_MEAN) / MODEL_STD
    tensor = torch.from_numpy(np.transpose(arr, (2, 0, 1)).copy()).unsqueeze(0)
    device = next(model.parameters()).device
    tensor = tensor.to(device)
    with torch.inference_mode():
        logits = model(tensor)
        probability = torch.sigmoid(logits)[0, 0].detach().cpu().numpy().astype(np.float32)
    probability = cv2.resize(probability, (tile_rgb.shape[1], tile_rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
    return np.clip(probability, 0.0, 1.0)


def run_tiled_building_inference(model, image, threshold=MODEL_THRESHOLD):
    """Sliding-window overlap inference using the trained PyTorch checkpoint."""
    import numpy as np

    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    h, w = rgb.shape[:2]
    input_h, input_w = _onnx_input_size(model)
    tile_h, tile_w = input_h, input_w
    stride_y = max(64, int(tile_h * 0.75))
    stride_x = max(64, int(tile_w * 0.75))
    prob_sum = np.zeros((h, w), dtype=np.float32)
    weight_sum = np.zeros((h, w), dtype=np.float32)
    ys = list(range(0, max(h - tile_h, 0) + 1, stride_y))
    xs = list(range(0, max(w - tile_w, 0) + 1, stride_x))
    if not ys or ys[-1] != max(h - tile_h, 0):
        ys.append(max(h - tile_h, 0))
    if not xs or xs[-1] != max(w - tile_w, 0):
        xs.append(max(w - tile_w, 0))
    for y in ys:
        for x in xs:
            y2, x2 = min(y + tile_h, h), min(x + tile_w, w)
            tile = rgb[y:y2, x:x2]
            prob = _run_building_tile(model, tile, input_h, input_w)
            prob_sum[y:y2, x:x2] += prob
            weight_sum[y:y2, x:x2] += 1.0
    probability = prob_sum / np.maximum(weight_sum, 1.0)
    return probability, probability >= float(threshold)


def choose_sane_mask(probability, preferred_threshold=0.65):
    """Choose a conservative threshold, raising it only when the mask is too broad."""
    import numpy as np

    # Never lower the requested threshold: this avoids accidentally accepting
    # weaker predictions when the candidate list is sorted numerically.
    candidates = [float(preferred_threshold)] + [
        t for t in (0.70, 0.75, 0.80, 0.85, 0.90, 0.94, 0.97, 0.99)
        if t > float(preferred_threshold)
    ]
    total = max(int(probability.size), 1)
    for threshold in candidates:
        mask = probability >= threshold
        coverage = float(mask.sum()) / total
        # These are only broad plausibility guards, not proof of correct detection.
        if 0.0005 <= coverage <= 0.35:
            return threshold, mask, coverage, threshold != float(preferred_threshold)

    return None, np.zeros(probability.shape, dtype=bool), 0.0, True

def image_to_data_url(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")

@app.route("/extract-buildings", methods=["POST"])
def extract_buildings():
    if "image" not in request.files:
        return jsonify({"error": "Please choose an aerial, drone, satellite, or building image."}), 400
    file = request.files["image"]
    if not file.filename:
        return jsonify({"error": "Please choose an image."}), 400
    allowed = {".jpg", ".jpeg", ".png", ".webp"}
    suffix = Path(file.filename).suffix.lower()
    if suffix not in allowed:
        return jsonify({"error": "Supported image types: JPG, JPEG, PNG, WEBP."}), 400

    safe_name = secure_filename(file.filename) or "upload"
    # Unique stored name prevents two uploads with the same filename overwriting each other.
    path = UPLOAD_DIR / f"ai_{uuid.uuid4().hex[:12]}_{safe_name}"

    try:
        file.save(path)
        import cv2
        model, _ = load_ai_model()
        # Apply camera EXIF orientation, then fully decode and validate the image.
        with Image.open(path) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
            image.load()
        if image.width < 64 or image.height < 64:
            return jsonify({"error": "Image is too small. Please upload an image at least 64 × 64 pixels."}), 400
        if image.width * image.height > 50_000_000:
            return jsonify({"error": "Image is too large for this prototype (maximum 50 megapixels). Resize it and try again."}), 413
        original = image.copy()

        # V8: tiled inference with a conservative threshold/coverage guard.
        # This reduces obvious saturated masks; visual review is still required.
        probability, initial_mask = run_tiled_building_inference(model, image, threshold=MODEL_THRESHOLD)
        threshold_used, mask, raw_coverage, threshold_adjusted = choose_sane_mask(probability, MODEL_THRESHOLD)
        total_pixels = int(mask.size)
        if threshold_used is None:
            pmin = float(np.nanmin(probability))
            pmax = float(np.nanmax(probability))
            pmean = float(np.nanmean(probability))
            return jsonify({
                "error": (
                    "AI mask rejected: the model did not produce a plausible building mask. "
                    "The output was nearly all-building or nearly empty even after conservative threshold checks. "
                    "No polygons were created. Try a clear, high-resolution top-down aerial image; if this repeats, "
                    "the model/preprocessing must be replaced or calibrated for this image source."
                ),
                "diagnostics": {"probability_min": round(pmin, 4), "probability_max": round(pmax, 4),
                                "probability_mean": round(pmean, 4), "initial_coverage_percent": round(float(initial_mask.mean()*100), 2)}
            }), 422
        building_pixels = int(mask.sum())
        coverage = 100.0 * building_pixels / max(total_pixels, 1)

        # Clean small pixel noise conservatively. Small kernels reduce the chance
        # that nearby roofs merge into one connected component.
        mask_u8 = (mask.astype(np.uint8) * 255)
        open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        clean = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, open_kernel)
        clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, close_kernel)

        # Fill enclosed holes, but keep the image border as background.
        h, w = clean.shape
        padded = cv2.copyMakeBorder(clean, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
        flood = padded.copy()
        flood_mask = np.zeros((h + 4, w + 4), np.uint8)
        cv2.floodFill(flood, flood_mask, (0, 0), 255)
        holes = cv2.bitwise_not(flood)
        clean = cv2.bitwise_or(padded, holes)[1:-1, 1:-1]

        # Component-wise quality filtering: remove tiny speckles and shapes that
        # are extremely thin or implausibly sparse. These filters are conservative
        # heuristics; they cannot replace a model trained for the local imagery.
        min_area = max(300, int(mask.size * 0.00030))
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            (clean > 0).astype(np.uint8), connectivity=8
        )
        accepted_mask = np.zeros_like(clean)
        polygons = []
        boundary = np.zeros_like(clean)
        rejected_components = 0

        for label_id in range(1, num_labels):
            x = int(stats[label_id, cv2.CC_STAT_LEFT])
            y = int(stats[label_id, cv2.CC_STAT_TOP])
            bw = int(stats[label_id, cv2.CC_STAT_WIDTH])
            bh = int(stats[label_id, cv2.CC_STAT_HEIGHT])
            component_area = int(stats[label_id, cv2.CC_STAT_AREA])
            if component_area < min_area or bw < 8 or bh < 8:
                rejected_components += 1
                continue

            component = (labels[y:y+bh, x:x+bw] == label_id).astype(np.uint8) * 255
            contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                rejected_components += 1
                continue
            contour = max(contours, key=cv2.contourArea)
            area_px = float(cv2.contourArea(contour))
            if area_px < min_area:
                rejected_components += 1
                continue

            perimeter = float(cv2.arcLength(contour, True))
            if perimeter <= 0:
                rejected_components += 1
                continue
            hull = cv2.convexHull(contour)
            hull_area = float(cv2.contourArea(hull))
            solidity = area_px / max(hull_area, 1.0)
            rectangularity = area_px / max(float(bw * bh), 1.0)
            aspect_ratio = max(bw, bh) / max(min(bw, bh), 1)
            mean_confidence = float(np.mean(probability[y:y+bh, x:x+bw][labels[y:y+bh, x:x+ bw] == label_id]))

            # Roofs can vary in shape, so these thresholds are intentionally broad.
            if solidity < 0.30 or rectangularity < 0.16 or aspect_ratio > 7.0 or mean_confidence < max(0.55, threshold_used - 0.01):
                rejected_components += 1
                continue

            # Polygon approximation keeps roof edges while avoiding excessive vertices.
            approx = cv2.approxPolyDP(contour, max(0.8, 0.0045 * perimeter), True)
            points = [[int(p[0][0] + x), int(p[0][1] + y)] for p in approx]
            if len(points) < 3:
                rejected_components += 1
                continue

            accepted_mask[labels == label_id] = 255
            polygons.append({
                "building_no": 0,
                "area_pixels": round(area_px, 1),
                "vertices": len(points),
                "bbox": {"x": x, "y": y, "width": bw, "height": bh},
                "polygon_pixels": points,
                "confidence_mean": round(mean_confidence, 4),
                "shape_solidity": round(solidity, 3),
                "shape_rectangularity": round(rectangularity, 3)
            })
            cv2.drawContours(boundary, [contour + np.array([[[x, y]]], dtype=np.int32)], -1, 255, 2)

        polygons.sort(key=lambda p: p["area_pixels"], reverse=True)
        for i, p in enumerate(polygons, 1):
            p["building_no"] = i

        # Metrics and visualizations use only components that passed the filters.
        clean = accepted_mask
        building_pixels = int((clean > 0).sum())
        coverage = 100.0 * building_pixels / max(total_pixels, 1)

        overlay = np.zeros((clean.shape[0], clean.shape[1], 4), dtype=np.uint8)
        overlay[clean > 0] = [255, 80, 80, 130]
        overlay_result = Image.alpha_composite(
            original.convert("RGBA"), Image.fromarray(overlay, "RGBA")
        ).convert("RGB")

        boundary_rgba = np.zeros((clean.shape[0], clean.shape[1], 4), dtype=np.uint8)
        boundary_rgba[boundary > 0] = [30, 120, 255, 255]
        polygon_result = Image.alpha_composite(
            original.convert("RGBA"), Image.fromarray(boundary_rgba, "RGBA")
        ).convert("RGB")

        footprint_arr = np.zeros((clean.shape[0], clean.shape[1], 3), dtype=np.uint8)
        footprint_arr[clean > 0] = [45, 105, 225]
        footprint = Image.fromarray(footprint_arr)

        if not polygons:
            return jsonify({
                "error": "No rooftop-like polygons passed the confidence and shape filters. Try a clear, top-down orthophoto without map labels/markers, or review the filtering thresholds in app.py.",
                "diagnostics": {
                    "threshold_used": round(float(threshold_used), 3),
                    "building_coverage_percent": round(coverage, 2),
                    "minimum_polygon_area_pixels": int(min_area),
                    "rejected_components": int(rejected_components)
                }
            }), 422

        return jsonify({
            "success": True,
            "model": "Trained WHU U-Net/ResNet34 PyTorch checkpoint + tiled inference + OpenCV polygon filters (V8 prototype)",
            "threshold_used": round(float(threshold_used), 3),
            "threshold_adjusted": bool(threshold_adjusted),
            "building_pixels": building_pixels,
            "image_pixels": total_pixels,
            "building_coverage_percent": round(coverage, 2),
            "building_classes": ["building"],
            "detected_buildings": len(polygons),
            "rejected_components": int(rejected_components),
            "diagnostics": {
                "probability_min": round(float(np.nanmin(probability)), 5),
                "probability_max": round(float(np.nanmax(probability)), 5),
                "probability_mean": round(float(np.nanmean(probability)), 5),
                "initial_threshold_coverage_percent": round(float(initial_mask.mean() * 100.0), 3),
                "final_accepted_coverage_percent": round(float(coverage), 3),
                "minimum_component_area_pixels": int(min_area),
                "accepted_components": int(len(polygons)),
                "rejected_components": int(rejected_components)
            },
            "polygons": polygons,
            "original": image_to_data_url(original),
            "overlay": image_to_data_url(overlay_result),
            "polygon_overlay": image_to_data_url(polygon_result),
            "footprint": image_to_data_url(footprint),
            "input_image_url": f"/uploads/{path.name}",
            "image_width": int(image.width),
            "image_height": int(image.height),
            "note": (
                f"{APP_VERSION} trained U-Net used probability threshold {threshold_used:.2f}" + (" (raised after an overly broad initial mask). " if threshold_adjusted else ". ") +
                f"Shape/confidence filters rejected {rejected_components} candidate components. "
                "These are candidate polygons, not verified building boundaries. Heuristics reduce some speckles but cannot guarantee roof-level accuracy. "
                "Use a clear orthophoto without map labels or markers and visually verify every polygon. "
                "Coordinates are image pixels; georeferencing, parcel matching and topology validation remain GIS tasks."
            )
        })
    except UnidentifiedImageError:
        return jsonify({"error": "The uploaded file is not a valid or readable image."}), 400
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    finally:
        # Keep uploaded image files available to the UI; they are served from /uploads.
        pass

@app.errorhandler(413)
def request_entity_too_large(_error):
    return jsonify({"error": "Upload is too large. Maximum upload size is 25 MB."}), 413

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001, debug=False)

import io
import math
import base64
import tempfile
import os

import cv2
import numpy as np
import streamlit as st
from PIL import Image


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Hand-Drawn Puppet Animator Pro",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🎨 Hand-Drawn Puppet Animator Pro")

st.markdown(
    """
Upload a hand-drawn character and automatically extract limbs and body parts based on color masks. 
The updated engine applies hierarchical puppet transformations and soft alpha blending for smooth motion.
"""
)


# ============================================================
# CONSTANTS
# ============================================================

MAX_IMAGE_SIZE = 1100

MOTIONS = [
    "Idle Breathing",
    "Wave",
    "Walk",
    "Run",
    "Jump",
    "Dance",
    "Bounce",
    "Shake",
    "Float",
    "Celebrate",
]


# ============================================================
# COLOR UTILITIES
# ============================================================

def get_color_name(rgb):
    r, g, b = [int(x) for x in rgb]
    pixel = np.uint8([[[b, g, r]]])
    hsv = cv2.cvtColor(pixel, cv2.COLOR_BGR2HSV)[0][0]

    hue = int(hsv[0])
    sat = int(hsv[1])

    if sat < 35:
        return "Neutral"

    if hue < 8 or hue >= 172:
        return "Red"
    elif hue < 22:
        return "Orange"
    elif hue < 35:
        return "Yellow"
    elif hue < 85:
        return "Green"
    elif hue < 130:
        return "Blue"
    elif hue < 155:
        return "Purple"
    else:
        return "Pink"


def extract_dominant_colors(image, num_clusters=8):
    try:
        small = cv2.resize(image, (160, 160), interpolation=cv2.INTER_AREA)
        hsv_small = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        saturation = hsv_small[:, :, 1]
        valid = saturation > 45

        pixels = small[valid].reshape(-1, 3)
        if len(pixels) < 50:
            return []

        pixels = pixels.astype(np.float32)
        k = min(num_clusters, max(2, len(pixels) // 80))

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1.0)
        _, labels, centers = cv2.kmeans(
            pixels, k, None, criteria, 5, cv2.KMEANS_PP_CENTERS
        )

        counts = np.bincount(labels.flatten(), minlength=k)
        total = max(1, int(np.sum(counts)))

        detected = []
        for idx, center in enumerate(centers):
            b, g, r = [int(x) for x in center]
            hsv = cv2.cvtColor(np.uint8([[[b, g, r]]]), cv2.COLOR_BGR2HSV)[0][0]

            h_val, s_val, v_val = int(hsv[0]), int(hsv[1]), int(hsv[2])
            coverage = counts[idx] / total * 100

            if coverage < 2.0 or s_val < 45:
                continue

            hex_code = f"#{r:02x}{g:02x}{b:02x}"
            detected.append(
                {
                    "label": f"{get_color_name((r, g, b))} ({hex_code}) {coverage:.1f}%",
                    "hsv": (h_val, s_val, v_val),
                    "hex": hex_code,
                    "rgb": (r, g, b),
                    "coverage": coverage,
                }
            )

        detected.sort(key=lambda x: x["coverage"], reverse=True)
        return detected
    except Exception:
        return []


# ============================================================
# COLOR MASKING & ENHANCED BLENDING
# ============================================================

def make_color_mask(image, color):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h0, s0, v0 = color["hsv"]
    hue_width = 18

    if h0 - hue_width < 0:
        lower1 = np.array([0, max(35, s0 - 90), max(30, v0 - 90)], dtype=np.uint8)
        upper1 = np.array([h0 + hue_width, 255, 255], dtype=np.uint8)
        lower2 = np.array([180 + h0 - hue_width, max(35, s0 - 90), max(30, v0 - 90)], dtype=np.uint8)
        upper2 = np.array([179, 255, 255], dtype=np.uint8)
        mask = cv2.bitwise_or(cv2.inRange(hsv, lower1, upper1), cv2.inRange(hsv, lower2, upper2))
    elif h0 + hue_width > 179:
        lower1 = np.array([h0 - hue_width, max(35, s0 - 90), max(30, v0 - 90)], dtype=np.uint8)
        upper1 = np.array([179, 255, 255], dtype=np.uint8)
        lower2 = np.array([0, max(35, s0 - 90), max(30, v0 - 90)], dtype=np.uint8)
        upper2 = np.array([(h0 + hue_width) - 180, 255, 255], dtype=np.uint8)
        mask = cv2.bitwise_or(cv2.inRange(hsv, lower1, upper1), cv2.inRange(hsv, lower2, upper2))
    else:
        lower = np.array([max(0, h0 - hue_width), max(35, s0 - 90), max(25, v0 - 90)], dtype=np.uint8)
        upper = np.array([min(179, h0 + hue_width), 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)

    kernel_small = np.ones((3, 3), np.uint8)
    kernel_medium = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_medium)

    return mask


def component_boundary(mask):
    if mask.dtype == bool:
        mask = mask.astype(np.uint8) * 255
    elif mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return np.empty((0, 2), dtype=np.float32)

    points = np.vstack([c.reshape(-1, 2) for c in contours])
    return points.astype(np.float32)


def farthest_point(points, origin):
    if len(points) == 0:
        return origin
    diff = points - np.array(origin, dtype=np.float32)
    dist = np.sum(diff * diff, axis=1)
    return tuple(points[np.argmax(dist)])


def closest_points_between_masks(mask_a, mask_b):
    pts_a = component_boundary(mask_a)
    pts_b = component_boundary(mask_b)

    if len(pts_a) == 0 or len(pts_b) == 0:
        return None, None, float("inf")

    max_points = 500
    if len(pts_a) > max_points:
        pts_a = pts_a[np.linspace(0, len(pts_a) - 1, max_points).astype(int)]
    if len(pts_b) > max_points:
        pts_b = pts_b[np.linspace(0, len(pts_b) - 1, max_points).astype(int)]

    diff = pts_a[:, None, :] - pts_b[None, :, :]
    dist2 = np.sum(diff * diff, axis=2)
    ia, ib = np.unravel_index(np.argmin(dist2), dist2.shape)

    return tuple(pts_a[ia]), tuple(pts_b[ib]), float(math.sqrt(dist2[ia, ib]))


def detect_parts(image, selected_colors):
    h, w = image.shape[:2]
    all_parts = []

    for color_index, color in enumerate(selected_colors):
        mask = make_color_mask(image, color)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        min_area = max(250, int(h * w * 0.00025))

        for i in range(1, num_labels):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area < min_area:
                continue

            component_mask = (labels == i).astype(np.uint8) * 255
            ys, xs = np.where(component_mask > 0)
            if len(xs) < 20:
                continue

            x, y, cw, ch = [int(v) for v in stats[i, :4]]
            cx, cy = centroids[i]

            boundary = component_boundary(component_mask)
            if len(boundary) == 0:
                continue

            pts = boundary.astype(np.float32)
            centered = pts - np.mean(pts, axis=0)
            covariance = np.cov(centered.T)
            eigenvalues, eigenvectors = np.linalg.eigh(covariance)

            axis = eigenvectors[:, np.argmax(eigenvalues)]
            projections = centered @ axis

            p_min = tuple(pts[np.argmin(projections)])
            p_max = tuple(pts[np.argmax(projections)])
            length = float(np.linalg.norm(np.array(p_max) - np.array(p_min)))

            all_parts.append(
                {
                    "id": len(all_parts),
                    "color_index": color_index,
                    "color_name": color["label"],
                    "mask": component_mask > 0,
                    "area": area,
                    "center": (float(cx), float(cy)),
                    "bbox": (x, y, cw, ch),
                    "base": (float(cx), float(cy)),
                    "tip": p_max,
                    "parent": None,
                    "depth": 0,
                }
            )

    if not all_parts:
        return []

    for part in all_parts:
        best_parent, best_distance, best_p1 = None, float("inf"), None
        for candidate in all_parts:
            if candidate["id"] == part["id"] or candidate["area"] < part["area"] * 0.30:
                continue

            p1, p2, distance = closest_points_between_masks(part["mask"], candidate["mask"])
            if p1 is None:
                continue

            tolerance = max(80, min(w, h) * 0.12)
            if distance < tolerance and distance < best_distance:
                best_distance = distance
                best_parent = candidate
                best_p1 = p1

        if best_parent is not None:
            part["parent"] = best_parent["id"]
            part["base"] = (float(best_p1[0]), float(best_p1[1]))
            child_boundary = component_boundary((part["mask"].astype(np.uint8) * 255))
            part["tip"] = farthest_point(child_boundary, part["base"])
        else:
            part["parent"] = None
            part["base"] = part["center"]
            part["tip"] = farthest_point(
                component_boundary((part["mask"].astype(np.uint8) * 255)), part["center"]
            )

    def get_depth(part):
        visited = set()
        current = part
        depth = 0
        while current["parent"] is not None and depth < 20:
            if current["id"] in visited:
                break
            visited.add(current["id"])
            parent = next((p for p in all_parts if p["id"] == current["parent"]), None)
            if parent is None:
                break
            current = parent
            depth += 1
        return depth

    for part in all_parts:
        part["depth"] = get_depth(part)

    all_parts.sort(key=lambda p: p["depth"])
    return all_parts


# ============================================================
# TRANSFORMATION ENGINE
# ============================================================

def get_affine_matrix(angle_deg, tx, ty, pivot):
    rad = math.radians(angle_deg)
    cos_a = math.cos(rad)
    sin_a = math.sin(rad)
    px, py = pivot

    M = np.array(
        [
            [cos_a, -sin_a, px - px * cos_a + py * sin_a + tx],
            [sin_a, cos_a, py - px * sin_a - py * cos_a + ty],
            [0, 0, 1],
        ],
        dtype=np.float32,
    )
    return M


def compute_world_transforms(parts, motion, t, intensity):
    transforms = {}
    for part in parts:
        angle, tx, ty = motion_for_part(motion, part, t, intensity)
        pivot = part["base"]

        local_M = get_affine_matrix(angle, tx, ty, pivot)
        parent_id = part["parent"]

        if parent_id is not None and parent_id in transforms:
            world_M = transforms[parent_id] @ local_M
        else:
            world_M = local_M

        transforms[part["id"]] = world_M

    return transforms


def prepare_background(original, parts):
    if not parts:
        return original.copy()

    combined = np.zeros(original.shape[:2], dtype=np.uint8)
    for part in parts:
        mask = part["mask"].astype(np.uint8) * 255
        mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=2)
        combined = cv2.bitwise_or(combined, mask)

    if (np.count_nonzero(combined) / float(combined.size)) > 0.55:
        return original.copy()

    try:
        return cv2.inpaint(original, combined, 5, cv2.INPAINT_TELEA)
    except Exception:
        return original.copy()


def paste_layer_soft(canvas, layer, layer_mask):
    """Anti-aliased soft edge compositor for seamless joint connections."""
    alpha = cv2.GaussianBlur(layer_mask, (5, 5), 0).astype(np.float32) / 255.0
    alpha = alpha[:, :, None]

    result = layer.astype(np.float32) * alpha + canvas.astype(np.float32) * (1.0 - alpha)
    return np.clip(result, 0, 255).astype(np.uint8)


# ============================================================
# MOTION CONTROLLERS
# ============================================================

def motion_for_part(motion, part, normalized_time, intensity):
    t = normalized_time
    sine = math.sin(2 * math.pi * t)
    side = 1 if part["id"] % 2 == 0 else -1
    depth = part.get("depth", 0)

    strength = intensity * (0.65 + min(depth, 3) * 0.15)
    angle, tx, ty = 0.0, 0.0, 0.0

    if motion == "Idle Breathing":
        angle = sine * strength * (0.10 if depth == 0 else 0.30)
        ty = sine * strength * 0.12 if depth == 0 else 0.0
    elif motion == "Wave":
        angle = sine * strength * (1.4 if depth >= 2 else 0.75)
        tx = sine * strength * 0.10
    elif motion == "Walk":
        phase = 2 * math.pi * t + (math.pi if side < 0 else 0)
        angle = math.sin(phase) * strength * 1.0 if depth > 0 else sine * strength * 0.12
        ty = (
            max(0, -math.cos(phase)) * strength * 0.18
            if depth > 0
            else abs(math.sin(2 * math.pi * t)) * strength * 0.18
        )
    elif motion == "Run":
        phase = 4 * math.pi * t + (math.pi if side < 0 else 0)
        angle = math.sin(phase) * strength * 1.5 if depth > 0 else sine * strength * 0.25
        ty = abs(math.sin(4 * math.pi * t)) * strength * 0.35
    elif motion == "Jump":
        ty = -math.sin(math.pi * t) * strength * 1.5
        angle = sine * strength * 0.30 if depth > 0 else 0.0
    elif motion == "Dance":
        angle = math.sin(4 * math.pi * t + depth * 0.5) * strength * 1.0
        tx = math.sin(2 * math.pi * t + depth) * strength * 0.25
        ty = math.cos(4 * math.pi * t) * strength * 0.20
    elif motion == "Bounce":
        ty = -abs(math.sin(math.pi * t)) * strength * 1.0
        angle = sine * strength * 0.35
    elif motion == "Shake":
        angle = math.sin(10 * math.pi * t) * strength * 0.55
        tx = math.sin(14 * math.pi * t) * strength * 0.25
    elif motion == "Float":
        ty = math.sin(2 * math.pi * t) * strength * 0.65
        tx = math.cos(2 * math.pi * t) * strength * 0.25
        angle = sine * strength * 0.30
    elif motion == "Celebrate":
        angle = sine * strength * 1.2
        ty = -abs(sine) * strength * 0.45
        tx = math.sin(4 * math.pi * t) * strength * 0.35 if depth > 0 else 0.0

    return angle, tx, ty


# ============================================================
# RENDERING & EXPORT
# ============================================================

def render_frame(original, background, parts, motion, t, intensity):
    canvas = background.copy()
    h, w = original.shape[:2]
    world_transforms = compute_world_transforms(parts, motion, t, intensity)

    for part in parts:
        mask = part["mask"].astype(np.uint8) * 255
        mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)

        affine_2x3 = world_transforms[part["id"]][:2, :]

        warped_part = cv2.warpAffine(
            original,
            affine_2x3,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255),
        )
        warped_mask = cv2.warpAffine(
            mask,
            affine_2x3,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

        canvas = paste_layer_soft(canvas, warped_part, warped_mask)

    return canvas


def build_gif(frames, duration):
    buffer = io.BytesIO()
    prepared = []
    for frame in frames:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        prepared.append(Image.fromarray(rgb).convert("P", palette=Image.ADAPTIVE))

    prepared[0].save(
        buffer,
        format="GIF",
        save_all=True,
        append_images=prepared[1:],
        duration=duration,
        loop=0,
    )
    return buffer.getvalue()


def build_mp4(frames, fps):
    if not frames:
        return None
    h, w = frames[0].shape[:2]
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        temp_path = tmp.name

    try:
        writer = cv2.VideoWriter(
            temp_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
        )
        for frame in frames:
            writer.write(frame)
        writer.release()

        with open(temp_path, "rb") as f:
            return f.read()
    except Exception:
        return None
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def draw_rig(image, parts):
    preview = image.copy()
    colors = [
        (0, 0, 255), (0, 180, 0), (255, 0, 0),
        (0, 180, 180), (180, 0, 180), (255, 120, 0),
    ]

    for index, part in enumerate(parts):
        color = colors[index % len(colors)]
        base = (int(part["base"][0]), int(part["base"][1]))
        tip = (int(part["tip"][0]), int(part["tip"][1]))
        center = (int(part["center"][0]), int(part["center"][1]))

        cv2.line(preview, base, tip, color, 3)
        cv2.circle(preview, base, 8, (0, 0, 255), -1)
        cv2.circle(preview, tip, 6, (0, 255, 0), -1)
        cv2.circle(preview, center, 4, color, -1)

        cv2.putText(
            preview, str(index + 1), (center[0] + 8, center[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA
        )

    return preview


# ============================================================
# MAIN APPLICATION INTERFACE
# ============================================================

uploaded = st.file_uploader(
    "Upload your hand-drawn character sheet", type=["jpg", "jpeg", "png", "webp"]
)

if uploaded is not None:
    try:
        file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if image is None:
            st.error("Could not read the uploaded image.")
            st.stop()

        h, w = image.shape[:2]
        if max(h, w) > MAX_IMAGE_SIZE:
            scale = MAX_IMAGE_SIZE / max(h, w)
            image = cv2.resize(
                image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA
            )

        st.sidebar.header("🎨 Part Detection")
        detected_colors = extract_dominant_colors(image)

        if not detected_colors:
            st.warning("No clear colored region detected.")
            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_container_width=True)
            st.stop()

        color_dict = {c["label"]: c for c in detected_colors}
        selected_labels = st.sidebar.multiselect(
            "Select body/part colors",
            list(color_dict.keys()),
            default=list(color_dict.keys())[:2],
        )

        selected_colors = [color_dict[label] for label in selected_labels]

        st.sidebar.header("🎬 Motion")
        motion = st.sidebar.selectbox("Animation Style", MOTIONS)
        intensity = st.sidebar.slider("Motion Strength", 2, 40, 14)
        fps = st.sidebar.select_slider("FPS", options=[8, 10, 12, 15, 20, 24, 30], value=12)
        duration_seconds = st.sidebar.slider("Duration (sec)", 1.0, 6.0, 2.0, 0.5)

        frame_count = max(8, int(fps * duration_seconds))
        parts = detect_parts(image, selected_colors)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("🖼️ Original Drawing")
            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_container_width=True)

        with col2:
            st.subheader(f"🦴 Puppet Rig ({len(parts)} Parts)")
            if parts:
                st.image(cv2.cvtColor(draw_rig(image, parts), cv2.COLOR_BGR2RGB), use_container_width=True)

        generate = st.button("✨ Generate Animation", type="primary", use_container_width=True)

        if generate and parts:
            progress = st.progress(0, text="Inpainting background...")

            # Simple Session Cache for Inpainted Background
            bg_key = f"bg_{uploaded.name}_{len(parts)}"
            if bg_key not in st.session_state:
                st.session_state[bg_key] = prepare_background(image, parts)
            background = st.session_state[bg_key]

            frames = []
            for i in range(frame_count):
                t = (i / max(1, frame_count - 1)) * 0.999
                frame = render_frame(image, background, parts, motion, t, intensity)
                frames.append(frame)
                progress.progress((i + 1) / frame_count, text=f"Rendering Frame {i + 1}/{frame_count}")

            progress.empty()

            gif_duration = int(1000 / fps)
            gif_data = build_gif(frames, gif_duration)

            st.subheader("🎬 Motion Preview")
            st.image(gif_data, use_container_width=True)

            d1, d2 = st.columns(2)
            safe_name = motion.lower().replace(" ", "_")

            with d1:
                st.download_button(
                    "⬇️ Download GIF",
                    data=gif_data,
                    file_name=f"{safe_name}.gif",
                    mime="image/gif",
                    use_container_width=True,
                )

            with d2:
                mp4_data = build_mp4(frames, fps)
                if mp4_data:
                    st.download_button(
                        "🎞️ Download MP4",
                        data=mp4_data,
                        file_name=f"{safe_name}.mp4",
                        mime="video/mp4",
                        use_container_width=True,
                    )

    except Exception as e:
        st.error(f"❌ Processing Error: {e}")
        st.exception(e)

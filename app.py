import io
import math
import os
import tempfile

import cv2
import numpy as np
import streamlit as st
from PIL import Image


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Hand-Drawn Character Merge & Reveal Animator",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🎨 Hand-Drawn Character Merge & Reveal Animator")

st.markdown(
    """
Animate one or two drawn characters walking in slowly from opposite directions, 
meeting at their original spots, and then cross-fading smoothly into your complete hand-drawn scene!
"""
)


# ============================================================
# CONSTANTS & UTILITIES
# ============================================================

MAX_IMAGE_SIZE = 1100

BACKGROUND_MODES = [
    "White / Light Paper",
    "Dark Background",
    "Automatic",
]


def resize_image(image, max_size=MAX_IMAGE_SIZE):
    h, w = image.shape[:2]
    if max(h, w) <= max_size:
        return image.copy()
    scale = max_size / float(max(h, w))
    return cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def bgr_to_rgb(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def ease_in_out(t):
    t = np.clip(t, 0.0, 1.0)
    return 0.5 - 0.5 * math.cos(math.pi * t)


# ============================================================
# BACKGROUND ESTIMATION & FOREGROUND DETECTION
# ============================================================

def estimate_background_lab(image):
    h, w = image.shape[:2]
    border_size = max(2, min(12, h // 4, w // 4))
    pixels = np.concatenate(
        [
            image[:border_size].reshape(-1, 3),
            image[h - border_size:].reshape(-1, 3),
            image[:, :border_size].reshape(-1, 3),
            image[:, w - border_size:].reshape(-1, 3),
        ],
        axis=0,
    )
    border_lab = cv2.cvtColor(pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2LAB).reshape(-1, 3)
    return np.median(border_lab, axis=0).astype(np.float32)


def detect_objects(image, sensitivity=50, background_mode="White / Light Paper"):
    h, w = image.shape[:2]
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    bg_lab = estimate_background_lab(image)

    diff = lab.astype(np.float32) - bg_lab.reshape(1, 1, 3)
    color_dist = np.sqrt(np.sum(diff * diff, axis=2))
    color_dist = (color_dist - color_dist.min()) / max(1e-6, (color_dist.max() - color_dist.min()))

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    bg_gray = np.median(gray)

    darkness = bg_gray - gray.astype(np.float32) if background_mode != "Dark Background" else gray.astype(np.float32) - bg_gray
    darkness = np.clip(darkness, 0, None)
    darkness = (darkness - darkness.min()) / max(1e-6, (darkness.max() - darkness.min()))

    score = cv2.GaussianBlur(color_dist * 0.6 + darkness * 0.4, (5, 5), 0)
    percentile = np.clip(88 - sensitivity * 0.35, 65, 90)
    threshold = np.percentile(score, percentile)

    mask = (score >= threshold).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    components = []
    min_area = max(25, int(h * w * 0.000025))

    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        x, y = int(stats[label, cv2.CC_STAT_LEFT]), int(stats[label, cv2.CC_STAT_TOP])
        cw, ch = int(stats[label, cv2.CC_STAT_WIDTH]), int(stats[label, cv2.CC_STAT_HEIGHT])
        cx, cy = centroids[label]
        components.append({"label": label, "area": area, "x": x, "y": y, "w": cw, "h": ch, "cx": float(cx), "cy": float(cy)})

    components.sort(key=lambda c: c["area"], reverse=True)
    return labels, components


def extract_single_character(image, labels, comp):
    mask = (labels == comp["label"]).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2)
    mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=1)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)

    x1, y1 = max(0, comp["x"] - 15), max(0, comp["y"] - 15)
    x2, y2 = min(image.shape[1], comp["x"] + comp["w"] + 15), min(image.shape[0], comp["y"] + comp["h"] + 15)

    char_crop = image[y1:y2, x1:x2].copy()
    alpha_crop = mask[y1:y2, x1:x2].copy()

    return {"crop": char_crop, "alpha": alpha_crop, "center": (comp["cx"], comp["cy"]), "w": x2 - x1, "h": y2 - y1}


def build_estimated_background(image, labels, selected_labels):
    mask = np.zeros(labels.shape, dtype=np.uint8)
    for lbl in selected_labels:
        mask[labels == lbl] = 255
    mask = cv2.dilate(mask, np.ones((9, 9), np.uint8), iterations=2)
    return cv2.inpaint(image, mask, 7, cv2.INPAINT_TELEA)


# ============================================================
# RENDERING ENGINE
# ============================================================

def transform_crop(crop, alpha, scale, angle):
    new_w, new_h = max(2, int(crop.shape[1] * scale)), max(2, int(crop.shape[0] * scale))
    resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    resized_alpha = cv2.resize(alpha, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    center = (new_w / 2.0, new_h / 2.0)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])

    bw, bh = max(2, int(new_h * sin + new_w * cos)), max(2, int(new_h * cos + new_w * sin))
    M[0, 2] += bw / 2 - center[0]
    M[1, 2] += bh / 2 - center[1]

    warped = cv2.warpAffine(resized, M, (bw, bh), borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    warped_a = cv2.warpAffine(resized_alpha, M, (bw, bh), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return warped, warped_a


def paste_crop(canvas, crop, alpha, cx, cy):
    ch, cw = crop.shape[:2]
    x1, y1 = int(round(cx - cw / 2.0)), int(round(cy - ch / 2.0))
    x2, y2 = x1 + cw, y1 + ch

    if x2 <= 0 or y2 <= 0 or x1 >= canvas.shape[1] or y1 >= canvas.shape[0]:
        return canvas

    cx1, cy1 = max(0, x1), max(0, y1)
    cx2, cy2 = min(canvas.shape[1], x2), min(canvas.shape[0], y2)

    src_x1, src_y1 = cx1 - x1, cy1 - y1
    src_x2, src_y2 = src_x1 + (cx2 - cx1), src_y1 + (cy2 - cy1)

    c_crop, a_crop = crop[src_y1:src_y2, src_x1:src_x2], alpha[src_y1:src_y2, src_x1:src_x2]
    a = (a_crop.astype(np.float32) / 255.0)[:, :, None]

    bg_crop = canvas[cy1:cy2, cx1:cx2].astype(np.float32)
    res = c_crop.astype(np.float32) * a + bg_crop * (1.0 - a)
    canvas[cy1:cy2, cx1:cx2] = np.clip(res, 0, 255).astype(np.uint8)
    return canvas


def render_combined_frame(
    original_img, estimated_bg, char_data_list, global_t, walk_in_frac, bob_amount, sway_amount, cycles
):
    h, w = original_img.shape[:2]
    canvas = estimated_bg.copy()

    fade_frac = 1.0 - walk_in_frac

    if global_t < walk_in_frac:
        # Phase 1: Walking In slowly toward home positions
        local_t = global_t / max(1e-6, walk_in_frac)
        movement = ease_in_out(local_t)

        phase = local_t * cycles * math.pi * 2
        bob = math.sin(phase) * bob_amount
        sway = math.sin(phase + math.pi / 2) * sway_amount

        for i, cdata in enumerate(char_data_list):
            # Calculate start positions outside canvas borders
            if len(char_data_list) == 2:
                start_x = -cdata["w"] if i == 0 else w + cdata["w"]
            else:
                start_x = -cdata["w"]
            start_y = cdata["center"][1]

            target_x, target_y = cdata["center"]
            cur_x = start_x + (target_x - start_x) * movement
            cur_y = start_y + (target_y - start_y) * movement

            warped_c, warped_a = transform_crop(cdata["crop"], cdata["alpha"], 1.0, sway)
            canvas = paste_crop(canvas, warped_c, warped_a, cur_x, cur_y + bob)

        return canvas

    else:
        # Phase 2: Stay at home spot & slowly cross-fade to original full image
        local_t = (global_t - walk_in_frac) / max(1e-6, fade_frac)
        fade_alpha = ease_in_out(local_t)

        for cdata in char_data_list:
            canvas = paste_crop(canvas, cdata["crop"], cdata["alpha"], cdata["center"][0], cdata["center"][1])

        # Blend canvas with the original hand-drawn image
        blended = canvas.astype(np.float32) * (1.0 - fade_alpha) + original_img.astype(np.float32) * fade_alpha
        return np.clip(blended, 0, 255).astype(np.uint8)


# ============================================================
# SIDEBAR CONTROLS
# ============================================================

st.sidebar.header("🎬 Slow Walk & Reveal Options")

fps = st.sidebar.select_slider("FPS", options=[8, 10, 12, 15, 20, 24], value=12)
duration = st.sidebar.slider("Animation Duration (sec)", 3.0, 12.0, 6.0, 0.5)

st.sidebar.markdown("---")
st.sidebar.header("🚶 Slow Walking Parameters")

walk_in_percent = st.sidebar.slider("Walk-In Duration (%)", 40, 80, 65)
bob_amount = st.sidebar.slider("Gait Bob (Vertical)", 0, 15, 4)
sway_amount = st.sidebar.slider("Gait Sway (Angle)", 0, 10, 2)
cycles = st.sidebar.slider("Total Walk Steps", 1, 10, 4)

st.sidebar.markdown("---")
st.sidebar.header("🎨 Extractions")
background_mode = st.sidebar.selectbox("Background Color Mode", BACKGROUND_MODES, index=0)
sensitivity = st.sidebar.slider("Extraction Sensitivity", 20, 80, 50)


# ============================================================
# MAIN FLOW
# ============================================================

uploaded = st.file_uploader("Upload your hand-drawn scene", type=["jpg", "jpeg", "png", "webp"])

if uploaded is not None:
    try:
        file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
        image = resize_image(cv2.imdecode(file_bytes, cv2.IMREAD_COLOR), MAX_IMAGE_SIZE)

        labels, components = detect_objects(image, sensitivity=sensitivity, background_mode=background_mode)

        if not components:
            st.error("❌ Could not isolate any drawn character.")
            st.stop()

        st.subheader("🎯 Choose Characters (Select 1 or 2)")
        st.caption("If 2 are chosen, they walk in from opposite sides and meet at their places before fading into the full scene.")

        options = [f"Object {i + 1} (Area {c['area']})" for i, c in enumerate(components[:10])]
        selected = st.multiselect("Select Character Objects", options, default=options[:min(2, len(options))])

        if not selected or len(selected) > 2:
            st.warning("Please select either 1 or 2 character objects.")
            st.stop()

        indices = [options.index(s) for s in selected]
        selected_components = [components[i] for i in indices]
        selected_labels = [c["label"] for c in selected_components]

        # Extract individual character crops
        char_data_list = [extract_single_character(image, labels, c) for c in selected_components]
        estimated_bg = build_estimated_background(image, labels, selected_labels)

        st.markdown("---")
        generate = st.button("✨ Generate Walk-In & Merge Animation", type="primary", use_container_width=True)

        if generate:
            frame_count = max(8, int(fps * duration))
            walk_in_frac = walk_in_percent / 100.0

            progress = st.progress(0, text="Rendering animation frames...")
            frames = []

            for i in range(frame_count):
                t = i / max(1, frame_count - 1)
                frame = render_combined_frame(
                    image, estimated_bg, char_data_list, t, walk_in_frac, bob_amount, sway_amount, cycles
                )
                frames.append(frame)
                progress.progress((i + 1) / frame_count)

            progress.empty()

            # GIF Compilation
            buffer = io.BytesIO()
            pil_frames = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)).convert("P", palette=Image.ADAPTIVE) for f in frames]
            pil_frames[0].save(
                buffer, format="GIF", save_all=True, append_images=pil_frames[1:], duration=int(1000 / fps), loop=0
            )

            st.subheader("🎬 Final Animation")
            st.image(buffer.getvalue(), use_container_width=True)

            st.download_button(
                "⬇️ Download Animation GIF",
                data=buffer.getvalue(),
                file_name="hand_drawn_merge_animation.gif",
                mime="image/gif",
                use_container_width=True,
            )

    except Exception as e:
        st.error(f"❌ Error during execution: {e}")
        st.exception(e)

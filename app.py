import io
import math
import os

import cv2
import numpy as np
import streamlit as st
from PIL import Image


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Interactive Hand-Drawn Character Animator",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🎨 Hand-Drawn Character Precise Separator & Animator")

st.markdown(
    """
Select your main character from complex hand-drawn artwork (ignoring grass, trees, and sky). 
The character will walk into the frame, settle into place, and seamlessly cross-fade into your complete drawing.
"""
)

MAX_IMAGE_SIZE = 1000


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def resize_image(image, max_size=MAX_IMAGE_SIZE):
    h, w = image.shape[:2]
    if max(h, w) <= max_size:
        return image.copy()
    scale = max_size / float(max(h, w))
    return cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def extract_paper_background(image):
    """Samples edge pixels to build a uniform background canvas."""
    h, w = image.shape[:2]
    border_pixels = np.concatenate([
        image[:15, :].reshape(-1, 3),
        image[-15:, :].reshape(-1, 3),
        image[:, :15].reshape(-1, 3),
        image[:, -15:].reshape(-1, 3)
    ], axis=0)
    bg_color = np.median(border_pixels, axis=0).astype(np.uint8)
    return np.full_like(image, bg_color)


def extract_character_interactive(image, bbox_pct):
    """
    Extracts the main character inside a user-defined percentage bounding box 
    using OpenCV GrabCut, removing surrounding scenery.
    """
    h, w = image.shape[:2]
    
    # Unpack percentage coordinates [x_min, y_min, x_max, y_max]
    xmin = int((bbox_pct[0] / 100.0) * w)
    ymin = int((bbox_pct[1] / 100.0) * h)
    xmax = int((bbox_pct[2] / 100.0) * w)
    ymax = int((bbox_pct[3] / 100.0) * h)
    
    rect_w = max(10, xmax - xmin)
    rect_h = max(10, ymax - ymin)
    
    rect = (xmin, ymin, rect_w, rect_h)
    
    # Initialize GrabCut mask
    gc_mask = np.zeros((h, w), np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    
    try:
        cv2.grabCut(image, gc_mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
        char_mask = np.where((gc_mask == 2) | (gc_mask == 0), 0, 255).astype(np.uint8)
    except Exception:
        # Fallback to simple rectangle crop if GrabCut fails
        char_mask = np.zeros((h, w), np.uint8)
        char_mask[ymin:ymax, xmin:xmax] = 255

    # Smooth edges
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    char_mask = cv2.morphologyEx(char_mask, cv2.MORPH_CLOSE, kernel)
    char_mask = cv2.GaussianBlur(char_mask, (3, 3), 0)

    # Crop out character
    ys, xs = np.where(char_mask > 20)
    if len(xs) == 0:
        return None, None, None

    x1, y1 = max(0, np.min(xs) - 5), max(0, np.min(ys) - 5)
    x2, y2 = min(w, np.max(xs) + 5), min(h, np.max(ys) + 5)

    char_crop = image[y1:y2, x1:x2].copy()
    alpha_crop = char_mask[y1:y2, x1:x2].copy()

    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0

    return char_crop, alpha_crop, (center_x, center_y)


# ============================================================
# RENDERING ENGINE
# ============================================================

def transform_crop(crop, alpha, angle):
    h, w = crop.shape[:2]
    center = (w / 2.0, h / 2.0)
    
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    
    bw, bh = max(2, int(h * sin + w * cos)), max(2, int(h * cos + w * sin))
    M[0, 2] += bw / 2 - center[0]
    M[1, 2] += bh / 2 - center[1]
    
    warped_c = cv2.warpAffine(crop, M, (bw, bh), borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    warped_a = cv2.warpAffine(alpha, M, (bw, bh), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return warped_c, warped_a


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


def ease_in_out(t):
    t = np.clip(t, 0.0, 1.0)
    return 0.5 - 0.5 * math.cos(math.pi * t)


def render_frame(original_img, paper_bg, char_crop, alpha_crop, home_center, global_t, walk_frac, bob_amt, sway_amt, cycles):
    canvas = paper_bg.copy()

    if global_t < walk_frac:
        # Phase 1: Slow walking from off-screen left to exact home position
        local_t = global_t / max(1e-6, walk_frac)
        movement = ease_in_out(local_t)

        start_x = -char_crop.shape[1]
        target_x, target_y = home_center

        cur_x = start_x + (target_x - start_x) * movement
        cur_y = target_y

        phase = local_t * cycles * math.pi * 2
        bob = math.sin(phase) * bob_amt
        sway = math.sin(phase + math.pi / 2) * sway_amt

        warped_c, warped_a = transform_crop(char_crop, alpha_crop, sway)
        canvas = paste_crop(canvas, warped_c, warped_a, cur_x, cur_y + bob)
        return canvas

    else:
        # Phase 2: Arrive at home position & cross-fade to reveal full drawing with all scenery
        local_t = (global_t - walk_frac) / max(1e-6, 1.0 - walk_frac)
        fade_alpha = ease_in_out(local_t)

        canvas = paste_crop(canvas, char_crop, alpha_crop, home_center[0], home_center[1])
        blended = canvas.astype(np.float32) * (1.0 - fade_alpha) + original_img.astype(np.float32) * fade_alpha
        return np.clip(blended, 0, 255).astype(np.uint8)


# ============================================================
# STREAMLIT UI
# ============================================================

st.sidebar.header("🎬 Motion Settings")
fps = st.sidebar.select_slider("FPS", options=[8, 10, 12, 15, 20, 24], value=12)
duration = st.sidebar.slider("Duration (sec)", 3.0, 10.0, 6.0, 0.5)

st.sidebar.markdown("---")
st.sidebar.header("🚶 Gait Settings")
walk_percent = st.sidebar.slider("Walk-In Duration (%)", 40, 80, 65)
bob_amount = st.sidebar.slider("Vertical Bobbing", 0, 20, 5)
sway_amount = st.sidebar.slider("Body Sway Angle", 0, 10, 3)
cycles = st.sidebar.slider("Walk Steps", 1, 10, 4)

uploaded = st.file_uploader("Upload Drawing", type=["jpg", "jpeg", "png", "webp"])

if uploaded is not None:
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    image = resize_image(cv2.imdecode(file_bytes, cv2.IMREAD_COLOR))

    st.subheader("🎯 Bounding Box Character Selector")
    st.caption("Adjust sliders so the red bounding box covers ONLY your main character (rabbit, boy, or family).")

    col_box1, col_box2 = st.columns(2)
    with col_box1:
        x_range = st.slider("Horizontal Range (X %)", 0, 100, (20, 80))
    with col_box2:
        y_range = st.slider("Vertical Range (Y %)", 0, 100, (10, 90))

    bbox_pct = [x_range[0], y_range[0], x_range[1], y_range[1]]

    # Draw live preview rectangle
    h, w = image.shape[:2]
    preview_img = image.copy()
    p_x1, p_y1 = int((bbox_pct[0] / 100.0) * w), int((bbox_pct[1] / 100.0) * h)
    p_x2, p_y2 = int((bbox_pct[2] / 100.0) * w), int((bbox_pct[3] / 100.0) * h)
    cv2.rectangle(preview_img, (p_x1, p_y1), (p_x2, p_y2), (0, 0, 255), 3)

    st.image(cv2.cvtColor(preview_img, cv2.COLOR_BGR2RGB), use_container_width=True)

    if st.button("✨ Extract Character & Animate", type="primary", use_container_width=True):
        with st.spinner("Extracting selected character with GrabCut..."):
            char_crop, alpha_crop, home_center = extract_character_interactive(image, bbox_pct)
            paper_bg = extract_paper_background(image)

        if char_crop is None:
            st.error("Could not extract character from selected region.")
            st.stop()

        frame_count = max(8, int(fps * duration))
        walk_frac = walk_percent / 100.0

        progress = st.progress(0, text="Rendering animation frames...")
        frames = []

        for i in range(frame_count):
            t = i / max(1, frame_count - 1)
            frame = render_frame(image, paper_bg, char_crop, alpha_crop, home_center, t, walk_frac, bob_amount, sway_amount, cycles)
            frames.append(frame)
            progress.progress((i + 1) / frame_count)

        progress.empty()

        # Build GIF
        buffer = io.BytesIO()
        pil_frames = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)).convert("P", palette=Image.ADAPTIVE) for f in frames]
        pil_frames[0].save(buffer, format="GIF", save_all=True, append_images=pil_frames[1:], duration=int(1000 / fps), loop=0)

        st.subheader("🎬 Final Animation")
        st.image(buffer.getvalue(), use_container_width=True)

        st.download_button(
            "⬇️ Download GIF",
            data=buffer.getvalue(),
            file_name="character_walk_in.gif",
            mime="image/gif",
            use_container_width=True,
        )

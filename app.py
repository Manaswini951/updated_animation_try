import io
import math
import os

import cv2
import numpy as np
import streamlit as st
from PIL import Image


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Main Character Only Animator",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🎨 Hand-Drawn Main Character Animator")

st.markdown(
    """
Isolate **only** your main character from a hand-drawn scene (leaving trees, clouds, and scenery untouched). 
The character walks smoothly from off-screen into its exact spot on the canvas, then blends back into the full original painting.
"""
)

MAX_IMAGE_SIZE = 1000


# ============================================================
# MAIN CHARACTER EXTRACTION (FILTERING SCENERY)
# ============================================================

def resize_image(image, max_size=MAX_IMAGE_SIZE):
    h, w = image.shape[:2]
    if max(h, w) <= max_size:
        return image.copy()
    scale = max_size / float(max(h, w))
    return cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def extract_paper_background(image):
    """Samples edge pixels to build a clean paper background for the walking path."""
    h, w = image.shape[:2]
    border_pixels = np.concatenate([
        image[:15, :].reshape(-1, 3),
        image[-15:, :].reshape(-1, 3),
        image[:, :15].reshape(-1, 3),
        image[:, -15:].reshape(-1, 3)
    ], axis=0)
    bg_color = np.median(border_pixels, axis=0).astype(np.uint8)
    return np.full_like(image, bg_color)


def extract_main_character_only(image):
    """
    Isolates only the primary subject (e.g. main animal/person) 
    and discards minor background scenery elements.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    
    # Threshold dark drawing lines
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 19, 3
    )
    
    # Threshold color saturation
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    _, sat_thresh = cv2.threshold(sat, 30, 255, cv2.THRESH_BINARY)
    
    combined_mask = cv2.bitwise_or(thresh, sat_thresh)
    
    # Close gaps in the main character outline
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    closed_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    
    # Find all connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(closed_mask)
    if num_labels <= 1:
        return None, None, None
        
    # Find the largest dominant object (ignoring background border at index 0)
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_idx = 1 + np.argmax(areas)
    
    # Build a mask ONLY for the largest character component
    main_char_mask = (labels == largest_idx).astype(np.uint8) * 255
    
    # Smooth edges for natural placement
    main_char_mask = cv2.GaussianBlur(main_char_mask, (5, 5), 0)
    
    ys, xs = np.where(main_char_mask > 20)
    if len(xs) == 0:
        return None, None, None
        
    x1, y1 = max(0, np.min(xs) - 5), max(0, np.min(ys) - 5)
    x2, y2 = min(image.shape[1], np.max(xs) + 5), min(image.shape[0], np.max(ys) + 5)
    
    char_crop = image[y1:y2, x1:x2].copy()
    alpha_crop = main_char_mask[y1:y2, x1:x2].copy()
    
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    
    return char_crop, alpha_crop, (center_x, center_y)


# ============================================================
# COMPOSITING & ANIMATION ENGINE
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
        # Phase 2: Arrive at home position & cross-fade to reveal full original scene (with all scenery)
        local_t = (global_t - walk_frac) / max(1e-6, 1.0 - walk_frac)
        fade_alpha = ease_in_out(local_t)

        canvas = paste_crop(canvas, char_crop, alpha_crop, home_center[0], home_center[1])
        blended = canvas.astype(np.float32) * (1.0 - fade_alpha) + original_img.astype(np.float32) * fade_alpha
        return np.clip(blended, 0, 255).astype(np.uint8)


# ============================================================
# STREAMLIT UI & CONTROLS
# ============================================================

st.sidebar.header("🎬 Motion Controls")
fps = st.sidebar.select_slider("FPS", options=[8, 10, 12, 15, 20, 24], value=12)
duration = st.sidebar.slider("Duration (seconds)", 3.0, 10.0, 6.0, 0.5)

st.sidebar.markdown("---")
st.sidebar.header("🚶 Gait Controls")
walk_percent = st.sidebar.slider("Walk-In Duration (%)", 40, 80, 65)
bob_amount = st.sidebar.slider("Vertical Bobbing", 0, 20, 5)
sway_amount = st.sidebar.slider("Body Sway Angle", 0, 10, 3)
cycles = st.sidebar.slider("Walk Steps", 1, 10, 4)

uploaded = st.file_uploader("Upload Hand-Drawn Painting", type=["jpg", "jpeg", "png", "webp"])

if uploaded is not None:
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    image = resize_image(cv2.imdecode(file_bytes, cv2.IMREAD_COLOR))

    with st.spinner("Extracting main character..."):
        char_crop, alpha_crop, home_center = extract_main_character_only(image)
        paper_bg = extract_paper_background(image)

    if char_crop is None:
        st.error("Could not isolate a distinct main character.")
        st.stop()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🖼️ Original Painting")
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_container_width=True)

    with col2:
        st.subheader("✂️ Separated Main Character Only")
        preview = paste_crop(paper_bg.copy(), char_crop, alpha_crop, char_crop.shape[1] // 2 + 10, char_crop.shape[0] // 2 + 10)
        st.image(cv2.cvtColor(preview, cv2.COLOR_BGR2RGB), use_container_width=True)

    if st.button("✨ Generate Walk-In & Merge Animation", type="primary", use_container_width=True):
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
            file_name="main_character_walk_in.gif",
            mime="image/gif",
            use_container_width=True,
        )

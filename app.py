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
    page_title="Multi-Image Character Separator & Animator",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🎨 Multi-Image Character Separator & Animator")

st.markdown(
    """
Upload one or multiple hand-drawn scenes. The script automatically handles orientation, 
extracts the main character from each drawing, and generates walk-in animations that cross-fade back into the full artwork.
"""
)

MAX_IMAGE_SIZE = 1000


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def auto_rotate_vertical(image):
    """Automatically rotates landscape/horizontal images vertically."""
    h, w = image.shape[:2]
    if w > h:
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return image


def resize_image(image, max_size=MAX_IMAGE_SIZE):
    h, w = image.shape[:2]
    if max(h, w) <= max_size:
        return image.copy()
    scale = max_size / float(max(h, w))
    return cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def extract_paper_background(image):
    """Samples edge pixels to create a clean canvas for movement."""
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
    """Extracts character using GrabCut inside percentage bounding box coordinates."""
    h, w = image.shape[:2]
    
    xmin = int((bbox_pct[0] / 100.0) * w)
    ymin = int((bbox_pct[1] / 100.0) * h)
    xmax = int((bbox_pct[2] / 100.0) * w)
    ymax = int((bbox_pct[3] / 100.0) * h)
    
    rect_w = max(10, xmax - xmin)
    rect_h = max(10, ymax - ymin)
    rect = (xmin, ymin, rect_w, rect_h)
    
    gc_mask = np.zeros((h, w), np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    
    try:
        cv2.grabCut(image, gc_mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
        char_mask = np.where((gc_mask == 2) | (gc_mask == 0), 0, 255).astype(np.uint8)
    except Exception:
        char_mask = np.zeros((h, w), np.uint8)
        char_mask[ymin:ymax, xmin:xmax] = 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    char_mask = cv2.morphologyEx(char_mask, cv2.MORPH_CLOSE, kernel)
    char_mask = cv2.GaussianBlur(char_mask, (3, 3), 0)

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

uploaded_files = st.file_uploader(
    "Upload Drawings (Select Multiple Files)", 
    type=["jpg", "jpeg", "png", "webp"],
    accept_multiple_files=True
)

if uploaded_files:
    st.info(f"📁 {len(uploaded_files)} file(s) loaded. Set the character bounding box area for batch extraction:")

    col_box1, col_box2 = st.columns(2)
    with col_box1:
        x_range = st.slider("Horizontal Range (X %)", 0, 100, (15, 85))
    with col_box2:
        y_range = st.slider("Vertical Range (Y %)", 0, 100, (10, 90))

    bbox_pct = [x_range[0], y_range[0], x_range[1], y_range[1]]

    if st.button("✨ Process & Animate All Drawings", type="primary", use_container_width=True):
        
        for idx, uploaded_file in enumerate(uploaded_files):
            st.markdown("---")
            st.subheader(f"🖼️ Drawing {idx + 1}: {uploaded_file.name}")

            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            raw_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            # Auto-rotate horizontal images to vertical
            image = auto_rotate_vertical(raw_image)
            image = resize_image(image)

            with st.spinner(f"Extracting character from {uploaded_file.name}..."):
                char_crop, alpha_crop, home_center = extract_character_interactive(image, bbox_pct)
                paper_bg = extract_paper_background(image)

            if char_crop is None:
                st.error(f"Could not extract character from {uploaded_file.name}.")
                continue

            frame_count = max(8, int(fps * duration))
            walk_frac = walk_percent / 100.0

            progress = st.progress(0, text=f"Rendering animation for {uploaded_file.name}...")
            frames = []

            for i in range(frame_count):
                t = i / max(1, frame_count - 1)
                frame = render_frame(image, paper_bg, char_crop, alpha_crop, home_center, t, walk_frac, bob_amount, sway_amount, cycles)
                frames.append(frame)
                progress.progress((i + 1) / frame_count)

            progress.empty()

            # Compile GIF
            buffer = io.BytesIO()
            pil_frames = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)).convert("P", palette=Image.ADAPTIVE) for f in frames]
            pil_frames[0].save(buffer, format="GIF", save_all=True, append_images=pil_frames[1:], duration=int(1000 / fps), loop=0)

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Processed Drawing**")
                st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_container_width=True)

            with c2:
                st.markdown("**Animation Preview**")
                st.image(buffer.getvalue(), use_container_width=True)

            st.download_button(
                f"⬇️ Download GIF ({uploaded_file.name})",
                data=buffer.getvalue(),
                file_name=f"animated_{uploaded_file.name}.gif",
                mime="image/gif",
                use_container_width=True,
            )

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
    page_title="Multi-Style Character Animator",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🎨 Hand-Drawn Multi-Style Character Animator")

st.markdown(
    """
Upload one or more hand-drawn artwork files. The app automatically detects orientation, 
orients characters vertically, extracts the main character, and applies your chosen animation style!
"""
)

MAX_IMAGE_SIZE = 1000
ANIMATION_STYLES = [
    "Walk-In & Settle",
    "Puppet Joint Tilt",
    "Pop-In Scale (Bounce)",
    "Float & Glide",
]


# ============================================================
# AUTOMATIC ROTATION & IMAGE PREPROCESSING
# ============================================================

def auto_rotate_vertical(image):
    """Detects horizontal orientation via aspect ratio and rotates to vertical."""
    h, w = image.shape[:2]
    
    # If wider than tall, rotate 90 degrees
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
    """Extracts the main character inside percentage bounding box coordinates using GrabCut."""
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
# ANIMATION STYLES & TRANSFORMS
# ============================================================

def transform_crop(crop, alpha, scale, angle, pivot_bottom=False):
    """Transforms image with scaling and rotation around center or bottom pivot point."""
    h, w = crop.shape[:2]
    
    if pivot_bottom:
        center = (w / 2.0, float(h))
    else:
        center = (w / 2.0, h / 2.0)
        
    scale = max(0.05, float(scale))
    new_w, new_h = max(2, int(w * scale)), max(2, int(h * scale))
    
    resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    resized_alpha = cv2.resize(alpha, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    pivot = (new_w / 2.0, new_h if pivot_bottom else new_h / 2.0)
    M = cv2.getRotationMatrix2D(pivot, angle, 1.0)
    
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    bw, bh = max(2, int(new_h * sin + new_w * cos)), max(2, int(new_h * cos + new_w * sin))
    
    M[0, 2] += bw / 2 - pivot[0]
    M[1, 2] += bh / 2 - pivot[1]
    
    warped_c = cv2.warpAffine(resized, M, (bw, bh), borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    warped_a = cv2.warpAffine(resized_alpha, M, (bw, bh), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    
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


def render_styled_frame(
    original_img, paper_bg, char_crop, alpha_crop, home_center, global_t, style, bob_amt, sway_amt, cycles
):
    canvas = paper_bg.copy()
    target_x, target_y = home_center

    if style == "Walk-In & Settle":
        walk_frac = 0.65
        if global_t < walk_frac:
            local_t = global_t / walk_frac
            movement = ease_in_out(local_t)
            cur_x = -char_crop.shape[1] + (target_x - (-char_crop.shape[1])) * movement
            phase = local_t * cycles * math.pi * 2
            bob = math.sin(phase) * bob_amt
            sway = math.sin(phase + math.pi / 2) * sway_amt
            warped_c, warped_a = transform_crop(char_crop, alpha_crop, 1.0, sway)
            return paste_crop(canvas, warped_c, warped_a, cur_x, target_y + bob)
        else:
            local_t = (global_t - walk_frac) / (1.0 - walk_frac)
            fade_alpha = ease_in_out(local_t)
            canvas = paste_crop(canvas, char_crop, alpha_crop, target_x, target_y)
            return np.clip(canvas.astype(np.float32) * (1.0 - fade_alpha) + original_img.astype(np.float32) * fade_alpha, 0, 255).astype(np.uint8)

    elif style == "Puppet Joint Tilt":
        # Tilts back and forth from bottom pivot point
        angle = math.sin(global_t * cycles * math.pi * 2) * (sway_amt * 2.5)
        bob = abs(math.sin(global_t * cycles * math.pi * 2)) * bob_amt
        warped_c, warped_a = transform_crop(char_crop, alpha_crop, 1.0, angle, pivot_bottom=True)
        canvas = paste_crop(canvas, warped_c, warped_a, target_x, target_y - bob)
        return canvas

    elif style == "Pop-In Scale (Bounce)":
        # Character pops up from size 0 to oversized, then settles
        pop_frac = 0.5
        if global_t < pop_frac:
            local_t = global_t / pop_frac
            scale = math.sin(local_t * math.pi * 0.8) * 1.25
        else:
            scale = 1.0
        angle = math.sin(global_t * math.pi * 2) * sway_amt
        warped_c, warped_a = transform_crop(char_crop, alpha_crop, scale, angle)
        return paste_crop(canvas, warped_c, warped_a, target_x, target_y)

    elif style == "Float & Glide":
        # Floating motion with subtle tilt
        ty = math.sin(global_t * math.pi * 2) * (bob_amt * 2.0)
        tx = math.cos(global_t * math.pi * 2) * (sway_amt * 1.5)
        angle = math.sin(global_t * math.pi * 2) * sway_amt
        warped_c, warped_a = transform_crop(char_crop, alpha_crop, 1.0, angle)
        return paste_crop(canvas, warped_c, warped_a, target_x + tx, target_y + ty)

    return canvas


# ============================================================
# STREAMLIT UI & BATCH PROCESSOR
# ============================================================

st.sidebar.header("🎬 Global Animation Controls")
style = st.sidebar.selectbox("Animation Style", ANIMATION_STYLES)
fps = st.sidebar.select_slider("FPS", options=[8, 10, 12, 15, 20, 24], value=12)
duration = st.sidebar.slider("Duration (sec)", 3.0, 10.0, 5.0, 0.5)

st.sidebar.markdown("---")
st.sidebar.header("⚙️ Fine Motion Adjustments")
bob_amount = st.sidebar.slider("Vertical Motion / Bobbing", 0, 25, 8)
sway_amount = st.sidebar.slider("Tilt / Sway Angle", 0, 20, 5)
cycles = st.sidebar.slider("Animation Loops / Steps", 1, 10, 3)

uploaded_files = st.file_uploader(
    "Upload Hand-Drawn Artwork (Multiple Files Supported)", 
    type=["jpg", "jpeg", "png", "webp"], 
    accept_multiple_files=True
)

if uploaded_files:
    st.info(f"📁 {len(uploaded_files)} file(s) uploaded. Adjust bounding box region below:")

    col_box1, col_box2 = st.columns(2)
    with col_box1:
        x_range = st.slider("Horizontal Range (X %)", 0, 100, (15, 85))
    with col_box2:
        y_range = st.slider("Vertical Range (Y %)", 0, 100, (10, 90))

    bbox_pct = [x_range[0], y_range[0], x_range[1], y_range[1]]

    if st.button("✨ Animate All Uploaded Images", type="primary", use_container_width=True):
        
        for idx, uploaded_file in enumerate(uploaded_files):
            st.markdown(f"---")
            st.subheader(f"🖼️ File {idx + 1}: {uploaded_file.name}")

            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            raw_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            # Auto-rotate to vertical if needed
            image = auto_rotate_vertical(raw_image)
            image = resize_image(image)

            with st.spinner("Extracting character..."):
                char_crop, alpha_crop, home_center = extract_character_interactive(image, bbox_pct)
                paper_bg = extract_paper_background(image)

            if char_crop is None:
                st.error(f"Could not extract character for {uploaded_file.name}.")
                continue

            frame_count = max(8, int(fps * duration))
            progress = st.progress(0, text=f"Rendering {style} frames...")
            frames = []

            for i in range(frame_count):
                t = i / max(1, frame_count - 1)
                frame = render_styled_frame(
                    image, paper_bg, char_crop, alpha_crop, home_center, t, style, bob_amount, sway_amount, cycles
                )
                frames.append(frame)
                progress.progress((i + 1) / frame_count)

            progress.empty()

            # Compile GIF
            buffer = io.BytesIO()
            pil_frames = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)).convert("P", palette=Image.ADAPTIVE) for f in frames]
            pil_frames[0].save(buffer, format="GIF", save_all=True, append_images=pil_frames[1:], duration=int(1000 / fps), loop=0)

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Processed Vertical Image**")
                st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_container_width=True)

            with c2:
                st.markdown(f"**{style} Preview**")
                st.image(buffer.getvalue(), use_container_width=True)

            st.download_button(
                f"⬇️ Download GIF ({uploaded_file.name})",
                data=buffer.getvalue(),
                file_name=f"animated_{uploaded_file.name}.gif",
                mime="image/gif",
                use_container_width=True,
            )

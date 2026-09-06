import io
import math
import os
import zipfile

import cv2
import numpy as np
import streamlit as st
from PIL import Image


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Advanced Hand-Drawn Color & Scene Animator",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🎨 Advanced Hand-Drawn Color & Scene Animator")

st.markdown(
    """
Select specific colors within your drawing to **wiggle, bounce, zoom, or glow** while staying 
naturally integrated with the scene. Export animations as standard videos or **transparent alpha GIFs/PNG sequences** for video editors!
"""
)

MAX_IMAGE_SIZE = 1000

ANIMATION_MODES = [
    "Seamless Wiggle & Sway",
    "Rhythmic Bounce & Stretch",
    "Glowing Zoom In/Out",
    "Storytelling Speech Cadence",
    "Walk-In & Merge with Original Scene",
]


# ============================================================
# IMAGE PREPROCESSING & COLOR UTILITIES
# ============================================================

def auto_rotate_vertical(image):
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


def get_color_name(rgb):
    r, g, b = [int(x) for x in rgb]
    pixel = np.uint8([[[b, g, r]]])
    hsv = cv2.cvtColor(pixel, cv2.COLOR_BGR2HSV)[0][0]
    hue, sat = int(hsv[0]), int(hsv[1])

    if sat < 30:
        return "Neutral/White/Gray"
    if hue < 10 or hue >= 170:
        return "Red"
    elif hue < 25:
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


def extract_dominant_colors(image, max_colors=6):
    """Identifies major saturated colors inside the artwork."""
    small = cv2.resize(image, (150, 150), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    
    # Filter out paper background/desaturated areas
    valid_pixels = small[hsv[:, :, 1] > 35].reshape(-1, 3)
    if len(valid_pixels) < 100:
        valid_pixels = small.reshape(-1, 3)

    pixels = valid_pixels.astype(np.float32)
    k = min(max_colors, max(2, len(pixels) // 100))
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 5, cv2.KMEANS_PP_CENTERS)
    
    counts = np.bincount(labels.flatten())
    total = sum(counts)

    detected = []
    for idx, center in enumerate(centers):
        b, g, r = [int(x) for x in center]
        hex_code = f"#{r:02x}{g:02x}{b:02x}"
        label = f"{get_color_name((r, g, b))} ({hex_code}) — {counts[idx]/total*100:.1f}%"
        detected.append({
            "label": label,
            "rgb": (r, g, b),
            "bgr": (b, g, r),
            "hex": hex_code,
            "coverage": counts[idx] / total
        })

    detected.sort(key=lambda x: x["coverage"], reverse=True)
    return detected


def make_color_mask(image, target_bgr, tolerance=45):
    """Creates a soft-edged mask for the selected color region."""
    diff = np.abs(image.astype(np.int16) - np.array(target_bgr, dtype=np.int16))
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    
    mask = (dist < tolerance).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.GaussianBlur(mask, (3, 3), 0)
    return mask


def extract_paper_background(image):
    h, w = image.shape[:2]
    border_pixels = np.concatenate([
        image[:15, :].reshape(-1, 3),
        image[-15:, :].reshape(-1, 3),
        image[:, :15].reshape(-1, 3),
        image[:, -15:].reshape(-1, 3)
    ], axis=0)
    bg_color = np.median(border_pixels, axis=0).astype(np.uint8)
    return np.full_like(image, bg_color)


# ============================================================
# ANIMATION & GLOW RENDERING ENGINE
# ============================================================

def transform_layer(crop, alpha, scale_x, scale_y, angle, pivot):
    h, w = crop.shape[:2]
    new_w = max(2, int(w * max(0.05, float(scale_x))))
    new_h = max(2, int(h * max(0.05, float(scale_y))))

    resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    resized_alpha = cv2.resize(alpha, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    px = (pivot[0] / float(w)) * new_w
    py = (pivot[1] / float(h)) * new_h

    M = cv2.getRotationMatrix2D((px, py), angle, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    bw, bh = max(2, int(new_h * sin + new_w * cos)), max(2, int(new_h * cos + new_w * sin))

    M[0, 2] += bw / 2 - px
    M[1, 2] += bh / 2 - py

    warped_c = cv2.warpAffine(resized, M, (bw, bh), borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    warped_a = cv2.warpAffine(resized_alpha, M, (bw, bh), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return warped_c, warped_a


def paste_layer(canvas, crop, alpha, cx, cy):
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


def apply_glow_effect(image, mask, intensity):
    """Creates an outer glowing aura around the animated region."""
    glow_mask = cv2.GaussianBlur(mask, (31, 31), 0).astype(np.float32) / 255.0
    glow_color = np.array([255, 235, 150], dtype=np.float32)  # Soft golden glow
    
    glow_layer = np.ones_like(image, dtype=np.float32) * glow_color
    alpha = (glow_mask * intensity)[:, :, None]
    
    result = image.astype(np.float32) * (1.0 - alpha * 0.5) + glow_layer * (alpha * 0.5)
    return np.clip(result, 0, 255).astype(np.uint8)


def render_color_animation_frame(
    original_img, mask, mode, global_t, speed, strength, transparent_bg=False
):
    h, w = original_img.shape[:2]
    ys, xs = np.where(mask > 20)

    if transparent_bg:
        # Create transparent RGBA canvas
        canvas = np.zeros((h, w, 4), dtype=np.uint8)
    else:
        canvas = original_img.copy()

    if len(xs) == 0:
        return canvas

    x1, y1, x2, y2 = np.min(xs), np.min(ys), np.max(xs), np.max(ys)
    crop = original_img[y1:y2, x1:x2].copy()
    alpha = mask[y1:y2, x1:x2].copy()
    
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    pivot = (cx - x1, cy - y1)
    
    phase = global_t * speed * math.pi * 2

    if mode == "Seamless Wiggle & Sway":
        angle = math.sin(phase) * strength
        scale_x, scale_y = 1.0, 1.0
        warped_c, warped_a = transform_layer(crop, alpha, scale_x, scale_y, angle, pivot)
        
    elif mode == "Rhythmic Bounce & Stretch":
        bounce = abs(math.sin(phase)) * strength * 0.5
        scale_y = 1.0 + math.sin(phase) * (strength * 0.02)
        scale_x = 1.0 - math.sin(phase) * (strength * 0.01)
        angle = math.sin(phase * 0.5) * (strength * 0.3)
        warped_c, warped_a = transform_layer(crop, alpha, scale_x, scale_y, angle, pivot)
        cy -= bounce

    elif mode == "Glowing Zoom In/Out":
        zoom = 1.0 + math.sin(phase) * (strength * 0.02)
        glow_val = (math.sin(phase) + 1.0) / 2.0 * (strength * 0.04)
        
        if not transparent_bg:
            canvas = apply_glow_effect(canvas, mask, glow_val)
            
        warped_c, warped_a = transform_layer(crop, alpha, zoom, zoom, 0.0, pivot)

    elif mode == "Storytelling Speech Cadence":
        angle = math.sin(phase * 0.5) * strength
        nod = abs(math.sin(phase * 2.0)) * strength * 0.8
        scale_y = 1.0 + math.sin(phase * 3.0) * 0.03
        scale_x = 1.0 - math.sin(phase * 3.0) * 0.02
        warped_c, warped_a = transform_layer(crop, alpha, scale_x, scale_y, angle, pivot)
        cy -= nod

    if transparent_bg:
        # Composite directly onto transparent RGBA canvas
        rgba_crop = cv2.cvtColor(warped_c, cv2.COLOR_BGR2BGRA)
        rgba_crop[:, :, 3] = warped_a
        return paste_layer(canvas, rgba_crop[:, :, :3], warped_a, cx, cy)
    else:
        return paste_layer(canvas, warped_c, warped_a, cx, cy)


# ============================================================
# EXPORT HELPERS (TRANSPARENT GIF & PNG ZIP)
# ============================================================

def build_transparent_gif(frames, fps):
    buffer = io.BytesIO()
    prepared = []
    duration = int(1000 / fps)

    for frame in frames:
        if frame.shape[2] == 4:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGBA)
            pil = Image.fromarray(rgb)
        else:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)
        prepared.append(pil)

    prepared[0].save(
        buffer, format="GIF", save_all=True, append_images=prepared[1:], duration=duration, loop=0, disposal=2
    )
    return buffer.getvalue()


# ============================================================
# STREAMLIT UI
# ============================================================

st.sidebar.header("🎬 Motion & Glow Settings")
animation_mode = st.sidebar.selectbox("Animation Style", ANIMATION_MODES)
fps = st.sidebar.select_slider("FPS", options=[8, 10, 12, 15, 20, 24], value=12)
duration = st.sidebar.slider("Duration (sec)", 2.0, 10.0, 5.0, 0.5)

st.sidebar.markdown("---")
st.sidebar.header("⚙️ Motion Adjustments")
speed = st.sidebar.slider("Animation Speed", 1, 8, 4)
strength = st.sidebar.slider("Motion / Zoom / Glow Intensity", 1, 20, 8)

st.sidebar.markdown("---")
st.sidebar.header("🎞️ Export Options")
transparent_bg = st.sidebar.checkbox("Export with Transparent Background (for Video Editing)", value=False)

uploaded_files = st.file_uploader(
    "Upload Drawings (Multiple Supported)", type=["jpg", "jpeg", "png", "webp"], accept_multiple_files=True
)

if uploaded_files:
    for idx, file in enumerate(uploaded_files):
        st.markdown("---")
        st.subheader(f"🖼️ Drawing {idx + 1}: {file.name}")

        file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
        raw_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        image = auto_rotate_vertical(raw_image)
        image = resize_image(image)

        # Detect dominant colors
        detected_colors = extract_dominant_colors(image)
        color_labels = [c["label"] for c in detected_colors]

        col1, col2 = st.columns(2)
        with col1:
            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Processed Image", use_container_width=True)

        with col2:
            st.markdown("**🎨 Select Color Region to Animate**")
            selected_label = st.selectbox("Identified Colors", color_labels, key=f"color_{idx}")
            selected_color = next(c for c in detected_colors if c["label"] == selected_label)
            tolerance = st.slider("Color Selection Area (Tolerance)", 10, 80, 45, key=f"tol_{idx}")

            mask = make_color_mask(image, selected_color["bgr"], tolerance)
            mask_preview = cv2.bitwise_and(image, image, mask=mask)
            st.image(cv2.cvtColor(mask_preview, cv2.COLOR_BGR2RGB), caption="Isolated Animated Part", use_container_width=True)

        if st.button(f"✨ Animate {file.name}", key=f"btn_{idx}", type="primary", use_container_width=True):
            frame_count = max(8, int(fps * duration))
            progress = st.progress(0, text="Rendering animation frames...")
            frames = []

            for i in range(frame_count):
                t = i / max(1, frame_count - 1)
                frame = render_color_animation_frame(
                    image, mask, animation_mode, t, speed, strength, transparent_bg
                )
                frames.append(frame)
                progress.progress((i + 1) / frame_count)

            progress.empty()

            gif_data = build_transparent_gif(frames, fps)

            st.subheader("🎬 Final Animated Preview")
            st.image(gif_data, use_container_width=True)

            st.download_button(
                f"⬇️ Download Animated GIF ({file.name})",
                data=gif_data,
                file_name=f"animated_{selected_color['hex']}_{file.name}.gif",
                mime="image/gif",
                use_container_width=True,
            )

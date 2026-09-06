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
    page_title="Complete Hand-Drawn Character & Color Animator",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🎨 Complete Hand-Drawn Character & Color Animator")

st.markdown(
    """
**Full Storytelling Pipeline:**
1. **Walk-In & Merge:** Character walks in, settles into place, and cross-fades into your full scene with all scenery intact.
2. **Color Region Animation:** Pick an identified color (e.g. shirt, hat, fur) to wiggle, bounce, talk, or zoom & glow while remaining attached to the image.
3. **Transparent Export:** Export as standard GIF or transparent RGBA stream for video editing software!
"""
)

MAX_IMAGE_SIZE = 1000

STAGE_CHOICES = [
    "1. Walk-In → Settle → Cross-Fade to Full Drawing",
    "2. Color Region Animation (Wiggle / Bounce / Glow / Talk)",
]

COLOR_ANIMATION_MODES = [
    "Seamless Wiggle & Sway",
    "Rhythmic Bounce & Stretch",
    "Glowing Zoom In/Out",
    "Storytelling Speech Cadence",
]


# ============================================================
# UTILITIES & COLOR EXTRACTION
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
    small = cv2.resize(image, (150, 150), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    
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
    diff = np.abs(image.astype(np.int16) - np.array(target_bgr, dtype=np.int16))
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    
    mask = (dist < tolerance).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.GaussianBlur(mask, (3, 3), 0)
    return mask


def extract_character_interactive(image, bbox_pct):
    h, w = image.shape[:2]
    xmin = int((bbox_pct[0] / 100.0) * w)
    ymin = int((bbox_pct[1] / 100.0) * h)
    xmax = int((bbox_pct[2] / 100.0) * w)
    ymax = int((bbox_pct[3] / 100.0) * h)
    
    rect = (xmin, ymin, max(10, xmax - xmin), max(10, ymax - ymin))
    
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

    return char_crop, alpha_crop, ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


# ============================================================
# RENDERING ENGINE
# ============================================================

def transform_layer(crop, alpha, scale_x, scale_y, angle, pivot):
    h, w = crop.shape[:2]
    new_w, new_h = max(2, int(w * max(0.05, float(scale_x)))), max(2, int(h * max(0.05, float(scale_y))))

    resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    resized_alpha = cv2.resize(alpha, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    px, py = (pivot[0] / float(w)) * new_w, (pivot[1] / float(h)) * new_h

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


def ease_in_out(t):
    t = np.clip(t, 0.0, 1.0)
    return 0.5 - 0.5 * math.cos(math.pi * t)


def apply_glow_effect(image, mask, intensity):
    glow_mask = cv2.GaussianBlur(mask, (31, 31), 0).astype(np.float32) / 255.0
    glow_layer = np.ones_like(image, dtype=np.float32) * np.array([255, 235, 150], dtype=np.float32)
    alpha = (glow_mask * intensity)[:, :, None]
    return np.clip(image.astype(np.float32) * (1.0 - alpha * 0.5) + glow_layer * (alpha * 0.5), 0, 255).astype(np.uint8)


# Stage 1: Walk-In & Merge
def render_walk_in_frame(original_img, paper_bg, char_crop, alpha_crop, home_center, global_t, walk_frac, bob_amt, sway_amt, cycles):
    canvas = paper_bg.copy()
    if global_t < walk_frac:
        local_t = global_t / max(1e-6, walk_frac)
        movement = ease_in_out(local_t)
        cur_x = -char_crop.shape[1] + (home_center[0] - (-char_crop.shape[1])) * movement
        phase = local_t * cycles * math.pi * 2
        bob = math.sin(phase) * bob_amt
        sway = math.sin(phase + math.pi / 2) * sway_amt
        warped_c, warped_a = transform_layer(char_crop, alpha_crop, 1.0, 1.0, sway, (char_crop.shape[1]/2, char_crop.shape[0]/2))
        return paste_layer(canvas, warped_c, warped_a, cur_x, home_center[1] + bob)
    else:
        local_t = (global_t - walk_frac) / max(1e-6, 1.0 - walk_frac)
        fade_alpha = ease_in_out(local_t)
        canvas = paste_layer(canvas, char_crop, alpha_crop, home_center[0], home_center[1])
        return np.clip(canvas.astype(np.float32) * (1.0 - fade_alpha) + original_img.astype(np.float32) * fade_alpha, 0, 255).astype(np.uint8)


# Stage 2: Color Animation
def render_color_animation_frame(original_img, mask, mode, global_t, speed, strength, transparent_bg):
    h, w = original_img.shape[:2]
    ys, xs = np.where(mask > 20)
    canvas = np.zeros((h, w, 4), dtype=np.uint8) if transparent_bg else original_img.copy()

    if len(xs) == 0:
        return canvas

    x1, y1, x2, y2 = np.min(xs), np.min(ys), np.max(xs), np.max(ys)
    crop, alpha = original_img[y1:y2, x1:x2].copy(), mask[y1:y2, x1:x2].copy()
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    pivot = (cx - x1, cy - y1)
    phase = global_t * speed * math.pi * 2

    if mode == "Seamless Wiggle & Sway":
        angle = math.sin(phase) * strength
        warped_c, warped_a = transform_layer(crop, alpha, 1.0, 1.0, angle, pivot)
    elif mode == "Rhythmic Bounce & Stretch":
        bounce = abs(math.sin(phase)) * strength * 0.5
        scale_y = 1.0 + math.sin(phase) * (strength * 0.02)
        scale_x = 1.0 - math.sin(phase) * (strength * 0.01)
        warped_c, warped_a = transform_layer(crop, alpha, scale_x, scale_y, math.sin(phase * 0.5) * strength * 0.3, pivot)
        cy -= bounce
    elif mode == "Glowing Zoom In/Out":
        zoom = 1.0 + math.sin(phase) * (strength * 0.02)
        if not transparent_bg:
            canvas = apply_glow_effect(canvas, mask, (math.sin(phase) + 1.0) / 2.0 * (strength * 0.04))
        warped_c, warped_a = transform_layer(crop, alpha, zoom, zoom, 0.0, pivot)
    elif mode == "Storytelling Speech Cadence":
        angle = math.sin(phase * 0.5) * strength
        nod = abs(math.sin(phase * 2.0)) * strength * 0.8
        warped_c, warped_a = transform_layer(crop, alpha, 1.0 - math.sin(phase * 3.0) * 0.02, 1.0 + math.sin(phase * 3.0) * 0.03, angle, pivot)
        cy -= nod

    if transparent_bg:
        rgba = cv2.cvtColor(warped_c, cv2.COLOR_BGR2BGRA)
        rgba[:, :, 3] = warped_a
        return paste_layer(canvas, rgba[:, :, :3], warped_a, cx, cy)
    else:
        return paste_layer(canvas, warped_c, warped_a, cx, cy)


def build_gif(frames, fps):
    buffer = io.BytesIO()
    prepared = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGRA2RGBA if f.shape[2] == 4 else cv2.COLOR_BGR2RGB)) for f in frames]
    prepared[0].save(buffer, format="GIF", save_all=True, append_images=prepared[1:], duration=int(1000 / fps), loop=0, disposal=2)
    return buffer.getvalue()


# ============================================================
# STREAMLIT UI
# ============================================================

st.sidebar.header("🎬 Pipeline Stage")
chosen_stage = st.sidebar.radio("Select Animation Phase", STAGE_CHOICES)

fps = st.sidebar.select_slider("FPS", options=[8, 10, 12, 15, 20, 24], value=12)
duration = st.sidebar.slider("Duration (sec)", 2.0, 10.0, 5.0, 0.5)

st.sidebar.markdown("---")

if "1. Walk-In" in chosen_stage:
    st.sidebar.header("🚶 Gait Controls")
    walk_percent = st.sidebar.slider("Walk-In Duration (%)", 40, 80, 65)
    bob_amount = st.sidebar.slider("Vertical Bobbing", 0, 20, 5)
    sway_amount = st.sidebar.slider("Body Sway Angle", 0, 10, 3)
    cycles = st.sidebar.slider("Walk Steps", 1, 10, 4)
else:
    st.sidebar.header("🎨 Color Motion Mode")
    color_mode = st.sidebar.selectbox("Color Motion Style", COLOR_ANIMATION_MODES)
    speed = st.sidebar.slider("Animation Speed", 1, 8, 4)
    strength = st.sidebar.slider("Motion / Zoom / Glow Intensity", 1, 20, 8)
    transparent_bg = st.sidebar.checkbox("Export with Transparent Background", value=False)

uploaded_files = st.file_uploader("Upload Drawings (Multiple Supported)", type=["jpg", "jpeg", "png", "webp"], accept_multiple_files=True)

if uploaded_files:
    for idx, file in enumerate(uploaded_files):
        st.markdown("---")
        st.subheader(f"🖼️ File {idx + 1}: {file.name}")

        file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
        image = resize_image(auto_rotate_vertical(cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)))

        if "1. Walk-In" in chosen_stage:
            col_box1, col_box2 = st.columns(2)
            with col_box1:
                x_range = st.slider(f"Horizontal Range (X %) #{idx+1}", 0, 100, (15, 85))
            with col_box2:
                y_range = st.slider(f"Vertical Range (Y %) #{idx+1}", 0, 100, (10, 90))

            bbox_pct = [x_range[0], y_range[0], x_range[1], y_range[1]]

            if st.button(f"✨ Generate Walk-In Animation ({file.name})", key=f"w_{idx}", type="primary", use_container_width=True):
                with st.spinner("Extracting character & background..."):
                    char_crop, alpha_crop, home_center = extract_character_interactive(image, bbox_pct)
                    paper_bg = extract_paper_background(image)

                frame_count = max(8, int(fps * duration))
                frames = [render_walk_in_frame(image, paper_bg, char_crop, alpha_crop, home_center, i/max(1, frame_count-1), walk_percent/100.0, bob_amount, sway_amount, cycles) for i in range(frame_count)]
                gif_data = build_gif(frames, fps)

                st.image(gif_data, use_container_width=True)
                st.download_button("⬇️ Download Walk-In GIF", gif_data, f"walk_in_{file.name}.gif", "image/gif", use_container_width=True)

        else:
            detected_colors = extract_dominant_colors(image)
            c1, c2 = st.columns(2)
            with c1:
                st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Processed Image", use_container_width=True)
            with c2:
                selected_label = st.selectbox("Identified Colors", [c["label"] for c in detected_colors], key=f"col_{idx}")
                selected_color = next(c for c in detected_colors if c["label"] == selected_label)
                tolerance = st.slider("Selection Tolerance", 10, 80, 45, key=f"tol_{idx}")
                mask = make_color_mask(image, selected_color["bgr"], tolerance)
                st.image(cv2.cvtColor(cv2.bitwise_and(image, image, mask=mask), cv2.COLOR_BGR2RGB), caption="Isolated Animated Part", use_container_width=True)

            if st.button(f"✨ Animate Color Region ({file.name})", key=f"c_{idx}", type="primary", use_container_width=True):
                frame_count = max(8, int(fps * duration))
                frames = [render_color_animation_frame(image, mask, color_mode, i/max(1, frame_count-1), speed, strength, transparent_bg) for i in range(frame_count)]
                gif_data = build_gif(frames, fps)

                st.image(gif_data, use_container_width=True)
                st.download_button("⬇️ Download Color Motion GIF", gif_data, f"color_motion_{selected_color['hex']}_{file.name}.gif", "image/gif", use_container_width=True)

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
    page_title="Hand-Drawn Storytelling Character Animator",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🎭 Storytelling & Dialogue Character Animator")

st.markdown(
    """
Animate your hand-drawn character with natural **speech dynamics, conversational leaning, head-nodding, and rhythmic storytelling gestures**!
"""
)

MAX_IMAGE_SIZE = 1000

STORY_STYLES = [
    "Expressive Storyteller (Nod & Sway)",
    "Excited Explainer (Speech Pulses)",
    "Thoughtful Ponderer (Lean & Pause)",
    "Conversational Bounce (Lively Talk)",
]


# ============================================================
# UTILITIES & EXTRACTION
# ============================================================

def auto_rotate_vertical(image):
    """Rotates horizontal drawings vertically."""
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


def extract_character_interactive(image, bbox_pct):
    h, w = image.shape[:2]
    xmin, ymin = int((bbox_pct[0] / 100.0) * w), int((bbox_pct[1] / 100.0) * h)
    xmax, ymax = int((bbox_pct[2] / 100.0) * w), int((bbox_pct[3] / 100.0) * h)
    
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
# STORYTELLING MOTION ENGINE
# ============================================================

def transform_crop(crop, alpha, scale_x, scale_y, angle, pivot_bottom=True):
    """Applies non-uniform scaling (squash & stretch) and rotation around a base joint pivot."""
    h, w = crop.shape[:2]
    
    new_w = max(2, int(w * max(0.05, float(scale_x))))
    new_h = max(2, int(h * max(0.05, float(scale_y))))
    
    resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    resized_alpha = cv2.resize(alpha, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    pivot = (new_w / 2.0, float(new_h) if pivot_bottom else new_h / 2.0)
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


def render_storytelling_frame(
    paper_bg, char_crop, alpha_crop, home_center, global_t, style, talk_speed, tilt_intensity, gesture_intensity
):
    canvas = paper_bg.copy()
    target_x, target_y = home_center
    
    phase = global_t * talk_speed * math.pi * 2

    if style == "Expressive Storyteller (Nod & Sway)":
        # Slow torso lean combined with rapid conversational head-nods
        body_angle = math.sin(phase * 0.3) * tilt_intensity
        head_nod = abs(math.sin(phase * 2.0)) * (gesture_intensity * 0.8)
        
        # Subtle speech-based squash & stretch
        scale_y = 1.0 + math.sin(phase * 3.0) * 0.03
        scale_x = 1.0 - math.sin(phase * 3.0) * 0.02
        
        warped_c, warped_a = transform_crop(char_crop, alpha_crop, scale_x, scale_y, body_angle, pivot_bottom=True)
        return paste_crop(canvas, warped_c, warped_a, target_x, target_y - head_nod)

    elif style == "Excited Explainer (Speech Pulses)":
        # High energy storytelling with speech accent pops
        accent_pulse = max(0.0, math.sin(phase * 2.5)) ** 3
        scale_y = 1.0 + accent_pulse * (gesture_intensity * 0.03)
        scale_x = 1.0 - accent_pulse * (gesture_intensity * 0.015)
        
        angle = math.sin(phase * 1.2) * (tilt_intensity * 1.5)
        bob = accent_pulse * gesture_intensity * 1.5
        
        warped_c, warped_a = transform_crop(char_crop, alpha_crop, scale_x, scale_y, angle, pivot_bottom=True)
        return paste_crop(canvas, warped_c, warped_a, target_x, target_y - bob)

    elif style == "Thoughtful Ponderer (Lean & Pause)":
        # Slow deliberate leaning with dramatic pauses
        raw_tilt = math.sin(phase * 0.5)
        angle = math.copysign(abs(raw_tilt) ** 0.5, raw_tilt) * tilt_intensity
        
        scale_y = 1.0 + math.cos(phase * 0.5) * 0.02
        scale_x = 1.0
        
        warped_c, warped_a = transform_crop(char_crop, alpha_crop, scale_x, scale_y, angle, pivot_bottom=True)
        return paste_crop(canvas, warped_c, warped_a, target_x, target_y)

    elif style == "Conversational Bounce (Lively Talk)":
        # Continuous jaw/body hop mixed with side-to-side speech swaying
        angle = math.sin(phase * 1.5) * tilt_intensity
        talk_hop = abs(math.sin(phase * 3.0)) * gesture_intensity
        
        # Quick vertical speech stretching
        scale_y = 1.0 + math.sin(phase * 3.0) * 0.04
        scale_x = 1.0 - math.sin(phase * 3.0) * 0.02
        
        warped_c, warped_a = transform_crop(char_crop, alpha_crop, scale_x, scale_y, angle, pivot_bottom=True)
        return paste_crop(canvas, warped_c, warped_a, target_x, target_y - talk_hop)

    return canvas


# ============================================================
# STREAMLIT UI
# ============================================================

st.sidebar.header("🗣️ Storytelling & Dialogue Controls")
style = st.sidebar.selectbox("Dialogue Style", STORY_STYLES)
fps = st.sidebar.select_slider("FPS", options=[8, 10, 12, 15, 20, 24], value=12)
duration = st.sidebar.slider("Story Duration (sec)", 3.0, 12.0, 6.0, 0.5)

st.sidebar.markdown("---")
st.sidebar.header("🎙️ Speech Expressiveness")
talk_speed = st.sidebar.slider("Speaking Cadence / Speed", 1, 8, 4)
tilt_intensity = st.sidebar.slider("Body Lean & Tilt Angle", 0, 20, 6)
gesture_intensity = st.sidebar.slider("Nod / Bounce Height", 0, 25, 10)

uploaded_files = st.file_uploader(
    "Upload Drawings (Multiple Supported)", 
    type=["jpg", "jpeg", "png", "webp"], 
    accept_multiple_files=True
)

if uploaded_files:
    st.info(f"📁 {len(uploaded_files)} file(s) uploaded. Adjust bounding region below:")

    col_box1, col_box2 = st.columns(2)
    with col_box1:
        x_range = st.slider("Horizontal Range (X %)", 0, 100, (15, 85))
    with col_box2:
        y_range = st.slider("Vertical Range (Y %)", 0, 100, (10, 90))

    bbox_pct = [x_range[0], y_range[0], x_range[1], y_range[1]]

    if st.button("✨ Animate Storytelling Dialogue", type="primary", use_container_width=True):
        
        for idx, uploaded_file in enumerate(uploaded_files):
            st.markdown("---")
            st.subheader(f"📖 Character {idx + 1}: {uploaded_file.name}")

            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            raw_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            image = auto_rotate_vertical(raw_image)
            image = resize_image(image)

            with st.spinner("Isolating character..."):
                char_crop, alpha_crop, home_center = extract_character_interactive(image, bbox_pct)
                paper_bg = extract_paper_background(image)

            if char_crop is None:
                st.error(f"Could not extract character for {uploaded_file.name}.")
                continue

            frame_count = max(8, int(fps * duration))
            progress = st.progress(0, text=f"Rendering {style}...")
            frames = []

            for i in range(frame_count):
                t = i / max(1, frame_count - 1)
                frame = render_storytelling_frame(
                    paper_bg, char_crop, alpha_crop, home_center, t, style, talk_speed, tilt_intensity, gesture_intensity
                )
                frames.append(frame)
                progress.progress((i + 1) / frame_count)

            progress.empty()

            buffer = io.BytesIO()
            pil_frames = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)).convert("P", palette=Image.ADAPTIVE) for f in frames]
            pil_frames[0].save(buffer, format="GIF", save_all=True, append_images=pil_frames[1:], duration=int(1000 / fps), loop=0)

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Character Cutout**")
                st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_container_width=True)

            with c2:
                st.markdown(f"**{style} Dialogue Animation**")
                st.image(buffer.getvalue(), use_container_width=True)

            st.download_button(
                f"⬇️ Download Story Animation ({uploaded_file.name})",
                data=buffer.getvalue(),
                file_name=f"story_dialogue_{uploaded_file.name}.gif",
                mime="image/gif",
                use_container_width=True,
            )

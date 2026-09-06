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
    page_title="Hand-Drawn Character & Precise Color Animator",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🎨 Hand-Drawn Character & Precise Color Animator")

st.markdown(
    """
**Fixed In This Update:**
1. **Background Glowing Fix:** Excludes paper/canvas colors from target animation selection so the background never wiggles or glows.
2. **Reliable Extraction:** Uses robust adaptive thresholding to prevent black frame outputs.
3. **Clean Subject Isolation:** Separates subject elements cleanly from background textures.
"""
)

MAX_IMAGE_SIZE = 1000

COLOR_ANIMATION_MODES = [
    "Seamless Wiggle & Sway",
    "Rhythmic Bounce & Stretch",
    "Glowing Zoom In/Out",
    "Storytelling Speech Cadence",
]


# ============================================================
# UTILITIES, ROTATION & WARMTH ENHANCEMENT
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


def enhance_color_temperature_and_warmth(image, temp_shift=12, saturation_boost=1.15):
    img_float = image.astype(np.float32)
    b, g, r = cv2.split(img_float)

    r = r * (1.0 + (temp_shift / 100.0))
    b = b * (1.0 - (temp_shift / 200.0))

    warmed = cv2.merge([b, g, r])
    warmed = np.clip(warmed, 0, 255).astype(np.uint8)

    hsv = cv2.cvtColor(warmed, cv2.COLOR_BGR2HSV).astype(np.float32)
    h, s, v = cv2.split(hsv)
    s = np.clip(s * saturation_boost, 0, 255)

    polished_hsv = cv2.merge([h, s, v]).astype(np.uint8)
    return cv2.cvtColor(polished_hsv, cv2.COLOR_HSV2BGR)


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


def inpaint_color_hole(image, mask):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    dilated_mask = cv2.dilate(mask, kernel, iterations=2)
    return cv2.inpaint(image, dilated_mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)


# ============================================================
# RELIABLE CHARACTER EXTRACTION
# ============================================================

def extract_character_interactive(image, bbox_pct):
    """
    Extracts foreground character with fallback adaptive thresholding 
    to prevent empty/black extraction outputs.
    """
    h, w = image.shape[:2]
    xmin = int((bbox_pct[0] / 100.0) * w)
    ymin = int((bbox_pct[1] / 100.0) * h)
    xmax = int((bbox_pct[2] / 100.0) * w)
    ymax = int((bbox_pct[3] / 100.0) * h)
    
    rect = (xmin, ymin, max(10, xmax - xmin), max(10, ymax - ymin))
    
    char_mask = np.zeros((h, w), np.uint8)
    gc_mask = np.zeros((h, w), np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    
    try:
        cv2.grabCut(image, gc_mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
        char_mask = np.where((gc_mask == 2) | (gc_mask == 0), 0, 255).astype(np.uint8)
    except Exception:
        char_mask = np.zeros((h, w), np.uint8)

    # Fallback to Adaptive Thresholding if GrabCut returns empty
    if np.count_nonzero(char_mask) < (rect[2] * rect[3] * 0.05):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        crop_gray = gray[ymin:ymax, xmin:xmax]
        
        # Adaptive Threshold to capture hand-drawn ink lines
        thresh = cv2.adaptiveThreshold(
            crop_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 10
        )
        
        # Fill inner holes using morphological close
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        char_mask[ymin:ymax, xmin:xmax] = closed

    # Isolate main foreground object contour
    contours, _ = cv2.findContours(char_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        main_contour = max(contours, key=cv2.contourArea)
        clean_mask = np.zeros_like(char_mask)
        cv2.drawContours(clean_mask, [main_contour], -1, 255, thickness=cv2.FILLED)
        char_mask = clean_mask

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    char_mask = cv2.morphologyEx(char_mask, cv2.MORPH_CLOSE, kernel)
    char_mask = cv2.GaussianBlur(char_mask, (3, 3), 0)

    ys, xs = np.where(char_mask > 20)
    if len(xs) == 0:
        # Ultimate Fallback: Default to bounding box center region
        char_mask[ymin:ymax, xmin:xmax] = 255
        ys, xs = np.where(char_mask > 20)

    x1, y1 = max(0, np.min(xs) - 5), max(0, np.min(ys) - 5)
    x2, y2 = min(w, np.max(xs) + 5), min(h, np.max(ys) + 5)

    char_crop = image[y1:y2, x1:x2].copy()
    alpha_crop = char_mask[y1:y2, x1:x2].copy()

    return char_crop, alpha_crop, ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


# ============================================================
# ACCURATE COLOR DISCRIMINATION (PREVENTS BACKGROUND SELECTION)
# ============================================================

def get_color_name(rgb):
    r, g, b = [int(x) for x in rgb]
    pixel = np.uint8([[[b, g, r]]])
    hsv = cv2.cvtColor(pixel, cv2.COLOR_BGR2HSV)[0][0]
    hue, sat, val = int(hsv[0]), int(hsv[1]), int(hsv[2])

    if sat < 30:
        return "Paper / Background Neutral"
    
    if hue < 10 or hue >= 170:
        return "Light Pink" if val > 180 and sat < 140 else "Dark Red / Pink"
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


def extract_dominant_colors(image, max_colors=8):
    small = cv2.resize(image, (150, 150), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    
    # Filter out paper/background low-saturation pixels
    valid_pixels = small[hsv[:, :, 1] > 35].reshape(-1, 3)
    if len(valid_pixels) < 50:
        valid_pixels = small.reshape(-1, 3)

    pixels = valid_pixels.astype(np.float32)
    k = min(max_colors, max(2, len(pixels) // 60))
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 25, 1.0)
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 5, cv2.KMEANS_PP_CENTERS)
    
    counts = np.bincount(labels.flatten())
    total = sum(counts)

    detected = []
    for idx, center in enumerate(centers):
        b, g, r = [int(x) for x in center]
        hex_code = f"#{r:02x}{g:02x}{b:02x}"
        c_name = get_color_name((r, g, b))
        
        # Exclude paper/background tones from primary list
        if "Paper" in c_name:
            continue

        label = f"{c_name} ({hex_code}) — {counts[idx]/total*100:.1f}%"
        detected.append({
            "label": label,
            "rgb": (r, g, b),
            "bgr": (b, g, r),
            "hex": hex_code,
            "coverage": counts[idx] / total
        })

    if not detected:
        # Fallback if no high-saturation colors exist
        for idx, center in enumerate(centers):
            b, g, r = [int(x) for x in center]
            hex_code = f"#{r:02x}{g:02x}{b:02x}"
            detected.append({
                "label": f"Color ({hex_code})",
                "rgb": (r, g, b),
                "bgr": (b, g, r),
                "hex": hex_code,
                "coverage": counts[idx] / total
            })

    detected.sort(key=lambda x: x["coverage"], reverse=True)
    return detected


def make_precise_color_mask(image, target_bgr, sharpness=25):
    """
    Creates an HSV mask for the selected color. 
    Applies safety limits to ensure background paper is never masked.
    """
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    target_pixel = np.uint8([[[target_bgr[0], target_bgr[1], target_bgr[2]]]])
    target_hsv = cv2.cvtColor(target_pixel, cv2.COLOR_BGR2HSV)[0][0]

    hue_tol = max(4, int(sharpness * 0.25))
    sat_tol = max(20, int(sharpness * 0.8))
    val_tol = max(20, int(sharpness * 0.8))

    # Force minimum saturation threshold to prevent matching paper background
    min_sat = max(35, int(target_hsv[1]) - sat_tol)

    lower_bound = np.array([
        max(0, int(target_hsv[0]) - hue_tol),
        min_sat,
        max(20, int(target_hsv[2]) - val_tol)
    ], dtype=np.uint8)

    upper_bound = np.array([
        min(179, int(target_hsv[0]) + hue_tol),
        min(255, int(target_hsv[1]) + sat_tol),
        min(255, int(target_hsv[2]) + val_tol)
    ], dtype=np.uint8)

    mask = cv2.inRange(hsv_image, lower_bound, upper_bound)

    # Safety Guard: If mask covers >45% of total image area, tighten bounds
    if np.count_nonzero(mask) > (image.shape[0] * image.shape[1] * 0.45):
        lower_bound[1] = max(60, lower_bound[1])
        mask = cv2.inRange(hsv_image, lower_bound, upper_bound)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


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

    if canvas.shape[2] == 4:
        if c_crop.shape[2] == 3:
            c_crop = cv2.cvtColor(c_crop, cv2.COLOR_BGR2BGRA)
            c_crop[:, :, 3] = a_crop
        result = c_crop.astype(np.float32) * a + bg_crop * (1.0 - a)
        result[:, :, 3] = np.maximum(bg_crop[:, :, 3], a_crop.astype(np.float32))
    else:
        if c_crop.shape[2] == 4:
            c_crop = c_crop[:, :, :3]
        result = c_crop.astype(np.float32) * a + bg_crop * (1.0 - a)

    canvas[cy1:cy2, cx1:cx2] = np.clip(result, 0, 255).astype(np.uint8)
    return canvas


def ease_in_out(t):
    t = np.clip(t, 0.0, 1.0)
    return 0.5 - 0.5 * math.cos(math.pi * t)


def apply_glow_effect(image, mask, intensity):
    glow_mask = cv2.GaussianBlur(mask, (31, 31), 0).astype(np.float32) / 255.0
    glow_layer = np.ones_like(image, dtype=np.float32) * np.array([255, 235, 150], dtype=np.float32)
    alpha = (glow_mask * intensity)[:, :, None]
    return np.clip(image.astype(np.float32) * (1.0 - alpha * 0.5) + glow_layer * (alpha * 0.5), 0, 255).astype(np.uint8)


def render_sequential_frame(
    original_img, paper_bg, char_crop, alpha_crop, home_center, color_mask, inpainted_bg,
    global_t, walk_frac, bob_amt, sway_amt, cycles, color_mode, speed, strength, transparent_mode=False
):
    h, w = original_img.shape[:2]

    # PHASE 1: WALK-IN FROM OFF-SCREEN
    if global_t < walk_frac:
        local_t = global_t / max(1e-6, walk_frac)
        movement = ease_in_out(local_t)
        cur_x = -char_crop.shape[1] + (home_center[0] - (-char_crop.shape[1])) * movement
        phase = local_t * cycles * math.pi * 2
        bob = math.sin(phase) * bob_amt
        sway = math.sin(phase + math.pi / 2) * sway_amt
        
        warped_c, warped_a = transform_layer(
            char_crop, alpha_crop, 1.0, 1.0, sway, (char_crop.shape[1] / 2.0, char_crop.shape[0] / 2.0)
        )

        if transparent_mode:
            canvas = np.zeros((h, w, 4), dtype=np.uint8)
            return paste_layer(canvas, warped_c, warped_a, cur_x, home_center[1] + bob)
        else:
            canvas = paper_bg.copy()
            return paste_layer(canvas, warped_c, warped_a, cur_x, home_center[1] + bob)

    # PHASE 2: CROSS-FADE TO ORIGINAL DRAWING & LOCALIZED COLOR MOTION
    else:
        local_t = (global_t - walk_frac) / max(1e-6, 1.0 - walk_frac)
        fade_alpha = ease_in_out(min(1.0, local_t * 2.5))

        if transparent_mode:
            base_img = np.zeros((h, w, 4), dtype=np.uint8)
            base_canvas = paste_layer(base_img, char_crop, alpha_crop, home_center[0], home_center[1])
        else:
            canvas_p1 = paste_layer(paper_bg.copy(), char_crop, alpha_crop, home_center[0], home_center[1])
            base_canvas = np.clip(
                canvas_p1.astype(np.float32) * (1.0 - fade_alpha) + inpainted_bg.astype(np.float32) * fade_alpha, 
                0, 255
            ).astype(np.uint8)

        ys, xs = np.where(color_mask > 20)
        if len(xs) == 0:
            return base_canvas

        x1, y1, x2, y2 = np.min(xs), np.min(ys), np.max(xs), np.max(ys)
        crop_c = original_img[y1:y2, x1:x2].copy()
        crop_a = color_mask[y1:y2, x1:x2].copy()
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        pivot = (cx - x1, cy - y1)
        phase_c = local_t * speed * math.pi * 2

        if color_mode == "Seamless Wiggle & Sway":
            angle = math.sin(phase_c) * strength
            warped_c, warped_a = transform_layer(crop_c, crop_a, 1.0, 1.0, angle, pivot)
        elif color_mode == "Rhythmic Bounce & Stretch":
            bounce = abs(math.sin(phase_c)) * strength * 0.5
            scale_y = 1.0 + math.sin(phase_c) * (strength * 0.02)
            scale_x = 1.0 - math.sin(phase_c) * (strength * 0.01)
            warped_c, warped_a = transform_layer(
                crop_c, crop_a, scale_x, scale_y, math.sin(phase_c * 0.5) * strength * 0.3, pivot
            )
            cy -= bounce
        elif color_mode == "Glowing Zoom In/Out":
            zoom = 1.0 + math.sin(phase_c) * (strength * 0.02)
            if not transparent_mode:
                base_canvas = apply_glow_effect(
                    base_canvas, color_mask, (math.sin(phase_c) + 1.0) / 2.0 * (strength * 0.04)
                )
            warped_c, warped_a = transform_layer(crop_c, crop_a, zoom, zoom, 0.0, pivot)
        elif color_mode == "Storytelling Speech Cadence":
            angle = math.sin(phase_c * 0.5) * strength
            nod = abs(math.sin(phase_c * 2.0)) * strength * 0.8
            warped_c, warped_a = transform_layer(
                crop_c, crop_a, 1.0 - math.sin(phase_c * 3.0) * 0.02, 1.0 + math.sin(phase_c * 3.0) * 0.03, angle, pivot
            )
            cy -= nod

        return paste_layer(base_canvas, warped_c, warped_a, cx, cy)


def build_gif(frames, fps):
    buffer = io.BytesIO()
    prepared = [
        Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGRA2RGBA if f.shape[2] == 4 else cv2.COLOR_BGR2RGB)) 
        for f in frames
    ]
    prepared[0].save(
        buffer, format="GIF", save_all=True, append_images=prepared[1:], duration=int(1000 / fps), loop=0, disposal=2
    )
    return buffer.getvalue()


# ============================================================
# STREAMLIT UI & CONTROLS
# ============================================================

st.sidebar.header("☀️ Warmth & Color Polish")
enable_enhancement = st.sidebar.checkbox("Enable Warm Temperature Polish", value=True)
temp_shift = st.sidebar.slider("Color Temperature Warmth", 0, 30, 12)

st.sidebar.markdown("---")
st.sidebar.header("🎬 Motion Controls")
fps = st.sidebar.select_slider("FPS", options=[8, 10, 12, 15, 20, 24], value=12)
duration = st.sidebar.slider("Total Duration (sec)", 4.0, 14.0, 7.0, 0.5)

st.sidebar.markdown("---")
st.sidebar.header("🚶 Gait Controls (Walk-In)")
walk_percent = st.sidebar.slider("Walk-In Duration (%)", 30, 70, 50)
bob_amount = st.sidebar.slider("Vertical Bobbing", 0, 20, 5)
sway_amount = st.sidebar.slider("Body Sway Angle", 0, 10, 3)
cycles = st.sidebar.slider("Walk Steps", 1, 10, 4)

st.sidebar.markdown("---")
st.sidebar.header("🎨 In-Scene Color Motion")
color_mode = st.sidebar.selectbox("Color Motion Style", COLOR_ANIMATION_MODES)
speed = st.sidebar.slider("Color Motion Speed", 1, 8, 4)
strength = st.sidebar.slider("Motion / Zoom / Glow Intensity", 1, 20, 8)

uploaded_files = st.file_uploader(
    "Upload Drawings (Select Multiple Files)", type=["jpg", "jpeg", "png", "webp"], accept_multiple_files=True
)

if uploaded_files:
    st.markdown("---")
    process_all = st.button("🚀 Process & Animate ALL Uploaded Files", type="primary", width="stretch")

    zip_export_files = {}

    for idx, file in enumerate(uploaded_files):
        st.markdown("---")
        st.subheader(f"🖼️ Drawing {idx + 1}: {file.name}")

        file.seek(0)
        file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
        raw_image = auto_rotate_vertical(cv2.imdecode(file_bytes, cv2.IMREAD_COLOR))

        if enable_enhancement:
            image = resize_image(enhance_color_temperature_and_warmth(raw_image, temp_shift=temp_shift))
        else:
            image = resize_image(raw_image)

        col_box1, col_box2 = st.columns(2)
        with col_box1:
            x_range = st.slider(f"Horizontal Bounding Box (X %) #{idx+1}", 0, 100, (15, 85), key=f"x_{idx}_{file.name}")
        with col_box2:
            y_range = st.slider(f"Vertical Bounding Box (Y %) #{idx+1}", 0, 100, (10, 90), key=f"y_{idx}_{file.name}")

        bbox_pct = [x_range[0], y_range[0], x_range[1], y_range[1]]

        detected_colors = extract_dominant_colors(image)
        c1, c2 = st.columns(2)
        with c1:
            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Processed Original Artwork", width="stretch")
        with c2:
            selected_label = st.selectbox(
                f"Identified Colors to Animate #{idx+1}", [c["label"] for c in detected_colors], key=f"col_{idx}_{file.name}"
            )
            selected_color = next(c for c in detected_colors if c["label"] == selected_label)
            sharpness = st.slider(f"Color Discrimination Sharpness #{idx+1}", 10, 60, 25, key=f"tol_{idx}_{file.name}")
            color_mask = make_precise_color_mask(image, selected_color["bgr"], sharpness)
            st.image(
                cv2.cvtColor(cv2.bitwise_and(image, image, mask=color_mask), cv2.COLOR_BGR2RGB), 
                caption="Isolated Color Region", width="stretch"
            )

        single_click = st.button(f"✨ Process & Animate ({file.name})", key=f"btn_{idx}_{file.name}", width="stretch")

        if process_all or single_click:
            with st.spinner(f"Processing {file.name}..."):
                char_crop, alpha_crop, home_center = extract_character_interactive(image, bbox_pct)
                paper_bg = extract_paper_background(image)
                inpainted_bg = inpaint_color_hole(image, color_mask)

            if char_crop is None:
                st.error(f"Could not extract character from {file.name}.")
                continue

            frame_count = max(8, int(fps * duration))
            walk_frac = walk_percent / 100.0

            progress = st.progress(0, text=f"Rendering Full Scene for {file.name}...")
            full_frames = []
            for i in range(frame_count):
                t = i / max(1, frame_count - 1)
                frame = render_sequential_frame(
                    image, paper_bg, char_crop, alpha_crop, home_center, color_mask, inpainted_bg,
                    t, walk_frac, bob_amount, sway_amount, cycles, color_mode, speed, strength, transparent_mode=False
                )
                full_frames.append(frame)
                progress.progress((i + 1) / frame_count)

            progress.empty()

            progress_trans = st.progress(0, text=f"Rendering Transparent Overlay for {file.name}...")
            transparent_frames = []
            for i in range(frame_count):
                t = i / max(1, frame_count - 1)
                frame_t = render_sequential_frame(
                    image, paper_bg, char_crop, alpha_crop, home_center, color_mask, inpainted_bg,
                    t, walk_frac, bob_amount, sway_amount, cycles, color_mode, speed, strength, transparent_mode=True
                )
                transparent_frames.append(frame_t)
                progress_trans.progress((i + 1) / frame_count)

            progress_trans.empty()

            full_gif = build_gif(full_frames, fps)
            trans_gif = build_gif(transparent_frames, fps)

            st.session_state[f"res_full_{idx}_{file.name}"] = full_gif
            st.session_state[f"res_trans_{idx}_{file.name}"] = trans_gif

        if f"res_full_{idx}_{file.name}" in st.session_state:
            full_gif = st.session_state[f"res_full_{idx}_{file.name}"]
            trans_gif = st.session_state[f"res_trans_{idx}_{file.name}"]

            zip_export_files[f"full_scene_{file.name}.gif"] = full_gif
            zip_export_files[f"transparent_overlay_{file.name}.gif"] = trans_gif

            st.subheader("🎬 Generated Previews & Downloads")
            p1, p2 = st.columns(2)
            with p1:
                st.markdown("**1. Full Scene Animated Sequence**")
                st.image(full_gif, width="stretch")
                st.download_button(
                    "⬇️ Download Full Scene GIF", full_gif, f"full_scene_{file.name}.gif", "image/gif", key=f"dl_full_{idx}_{file.name}", width="stretch"
                )
            with p2:
                st.markdown("**2. Transparent Overlay Sequence (For Video Editors)**")
                st.image(trans_gif, width="stretch")
                st.download_button(
                    "⬇️ Download Transparent GIF", trans_gif, f"transparent_overlay_{file.name}.gif", "image/gif", key=f"dl_trans_{idx}_{file.name}", width="stretch"
                )

    if zip_export_files:
        st.markdown("---")
        st.subheader("📦 Bulk Download All Generated Animations")
        
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
            for file_name, file_bytes in zip_export_files.items():
                zip_file.writestr(file_name, file_bytes)

        st.download_button(
            "📦 Download All Animations (ZIP Archive)",
            data=zip_buffer.getvalue(),
            file_name="all_animated_drawings.zip",
            mime="application/zip",
            type="primary",
            width="stretch",
        )

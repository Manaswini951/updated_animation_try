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
    page_title="Complete Unified Hand-Drawn Animator",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🎨 Complete Unified Hand-Drawn Animator")

st.markdown(
    """
**Sequential Animation & Extraction Pipeline:**
1. **Clean Extraction:** Isolates character outline cleanly without grabbing paper textures or scenery.
2. **Smooth Walk-In & Merge:** Character walks in smoothly, settles into place, and cross-fades into the complete scene.
3. **Targeted In-Scene Color Motion:** Wiggles/bounces ONLY the selected color pixels with HSV discrimination.
4. **Dual Export & Bulk ZIP:** Generates both Full Scene and Transparent Overlay GIFs, plus a bulk ZIP download.
"""
)

MAX_IMAGE_SIZE = 1100

COLOR_ANIMATION_MODES = [
    "Seamless Wiggle & Sway",
    "Rhythmic Bounce & Stretch",
    "Glowing Zoom In/Out",
    "Storytelling Speech Cadence",
]


# ============================================================
# IMAGE UTILITIES & ENHANCEMENTS
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
    return cv2.resize(
        image,
        (max(2, int(w * scale)), max(2, int(h * scale))),
        interpolation=cv2.INTER_AREA,
    )


def enhance_color_temperature_and_warmth(image, temp_shift=6, saturation_boost=1.03):
    img = image.astype(np.float32)
    b, g, r = cv2.split(img)

    r *= 1.0 + temp_shift / 220.0
    b *= 1.0 - temp_shift / 330.0

    warmed = cv2.merge([b, g, r])
    warmed = np.clip(warmed, 0, 255).astype(np.uint8)

    hsv = cv2.cvtColor(warmed, cv2.COLOR_BGR2HSV).astype(np.float32)
    h, s, v = cv2.split(hsv)
    s = np.clip(s * saturation_boost, 0, 255)

    polished = cv2.merge([h, s, v]).astype(np.uint8)
    return cv2.cvtColor(polished, cv2.COLOR_HSV2BGR)


# ============================================================
# EXTRACTION & OUTLINE DETECTION
# ============================================================

def bbox_from_percentages(image, bbox_pct):
    h, w = image.shape[:2]
    x1 = int(np.clip(bbox_pct[0], 0, 100) * w / 100.0)
    y1 = int(np.clip(bbox_pct[1], 0, 100) * h / 100.0)
    x2 = int(np.clip(bbox_pct[2], 0, 100) * w / 100.0)
    y2 = int(np.clip(bbox_pct[3], 0, 100) * h / 100.0)

    x1, x2 = sorted([x1, x2])
    y1, y2 = sorted([y1, y2])

    x2 = min(max(x2, x1 + 5), w)
    y2 = min(max(y2, y1 + 5), h)
    return x1, y1, x2, y2


def detect_dark_outline(crop, sensitivity=55):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    blur_size = int(np.clip(min(h, w) * 0.07, 15, 71))
    if blur_size % 2 == 0:
        blur_size += 1

    local_bg = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)
    darkness = cv2.subtract(local_bg, gray)

    darkness_percentile = np.interp(sensitivity, [0, 100], [97, 82])
    threshold_dark = np.percentile(darkness, darkness_percentile)

    local_mask = np.where(darkness >= threshold_dark, 255, 0).astype(np.uint8)

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    absolute_value_threshold = int(np.interp(sensitivity, [0, 100], [105, 170]))
    absolute_dark = (value <= absolute_value_threshold) & ((saturation >= 10) | (value <= 125))
    absolute_dark = absolute_dark.astype(np.uint8) * 255

    ink = cv2.bitwise_or(local_mask, absolute_dark)
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    ink = cv2.morphologyEx(ink, cv2.MORPH_OPEN, open_kernel, iterations=1)

    close_size = int(np.interp(sensitivity, [0, 100], [5, 11]))
    if close_size % 2 == 0:
        close_size += 1
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size))
    return cv2.morphologyEx(ink, cv2.MORPH_CLOSE, close_kernel, iterations=1)


def contour_fill_from_outline(outline, sensitivity=55):
    h, w = outline.shape[:2]
    contours, hierarchy = cv2.findContours(outline, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros_like(outline)

    crop_area = float(h * w)
    cx0, cy0 = w / 2.0, h / 2.0
    candidates = []

    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area < crop_area * 0.003:
            continue

        x, y, cw, ch = cv2.boundingRect(contour)
        center_x, center_y = x + cw / 2.0, y + ch / 2.0
        distance = math.sqrt(((center_x - cx0) / max(1, w)) ** 2 + ((center_y - cy0) / max(1, h)) ** 2)

        centre_score = max(0.0, 1.0 - distance * 2.5)
        size_score = min(1.0, area / (crop_area * 0.30))
        score = area * (0.55 + 0.30 * centre_score + 0.15 * size_score)

        candidates.append((score, i))

    if not candidates:
        return np.zeros_like(outline)

    candidates.sort(key=lambda item: item[0], reverse=True)
    best_index = candidates[0][1]

    silhouette = np.zeros_like(outline)
    cv2.drawContours(silhouette, contours, best_index, 255, thickness=cv2.FILLED)

    bridge_size = int(np.interp(sensitivity, [0, 100], [3, 7]))
    if bridge_size % 2 == 0:
        bridge_size += 1
    bridge_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (bridge_size, bridge_size))
    return cv2.morphologyEx(silhouette, cv2.MORPH_CLOSE, bridge_kernel, iterations=1)


def fallback_grabcut(crop, outline):
    h, w = crop.shape[:2]
    mask = np.full((h, w), cv2.GC_PR_BGD, dtype=np.uint8)
    border = max(2, int(min(h, w) * 0.025))

    mask[:border, :] = cv2.GC_BGD
    mask[-border:, :] = cv2.GC_BGD
    mask[:, :border] = cv2.GC_BGD
    mask[:, -border:] = cv2.GC_BGD

    mask[outline > 0] = cv2.GC_PR_FGD
    seed = cv2.dilate(outline, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)), iterations=1)
    mask[seed > 0] = cv2.GC_PR_FGD

    bg_model = np.zeros((1, 65), np.float64)
    fg_model = np.zeros((1, 65), np.float64)

    try:
        cv2.grabCut(crop, mask, None, bg_model, fg_model, 5, cv2.GC_INIT_WITH_MASK)
        result = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
        
        n, labels, stats, _ = cv2.connectedComponentsWithStats(result, connectivity=8)
        if n <= 1:
            return np.zeros_like(outline)
        areas = stats[1:, cv2.CC_STAT_AREA]
        index = 1 + int(np.argmax(areas))
        return np.where(labels == index, 255, 0).astype(np.uint8)
    except Exception:
        return np.zeros_like(outline)


def extract_character_mask(image, bbox_pct, outline_sensitivity=55, detail_strength=45):
    x1, y1, x2, y2 = bbox_from_percentages(image, bbox_pct)
    crop = image[y1:y2, x1:x2].copy()
    ch, cw = crop.shape[:2]

    if ch < 10 or cw < 10:
        return np.zeros(image.shape[:2], dtype=np.uint8), (x1, y1, x2, y2)

    outline = detect_dark_outline(crop, sensitivity=outline_sensitivity)
    silhouette = contour_fill_from_outline(outline, sensitivity=outline_sensitivity)

    silhouette_area = np.count_nonzero(silhouette)
    bbox_area = max(1, cw * ch)
    if silhouette_area < bbox_area * 0.01 or silhouette_area > bbox_area * 0.88:
        silhouette = fallback_grabcut(crop, outline)

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1].astype(np.float32)
    value = hsv[:, :, 2].astype(np.float32)

    colour_pixels = ((saturation > np.interp(detail_strength, [0, 100], [22, 10])) & (value < 252)).astype(np.uint8) * 255

    proximity_size = int(np.clip(min(ch, cw) * 0.055, 9, 31))
    if proximity_size % 2 == 0:
        proximity_size += 1
    proximity_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (proximity_size, proximity_size))

    silhouette_neighbourhood = cv2.dilate(silhouette, proximity_kernel, iterations=1)
    nearby_details = cv2.bitwise_and(colour_pixels, silhouette_neighbourhood)

    combined = cv2.bitwise_or(silhouette, nearby_details)
    combined = np.where(combined > 0, 255, 0).astype(np.uint8)

    full_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    full_mask[y1:y2, x1:x2] = combined
    return cv2.GaussianBlur(full_mask, (3, 3), 0), (x1, y1, x2, y2)


def create_background_plate(image, character_mask):
    hard_mask = np.where(character_mask > 30, 255, 0).astype(np.uint8)
    h, w = hard_mask.shape

    dilation_size = int(np.clip(min(h, w) * 0.012, 5, 25))
    if dilation_size % 2 == 0:
        dilation_size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_size, dilation_size))
    removal_mask = cv2.dilate(hard_mask, kernel, iterations=1)

    try:
        plate = cv2.inpaint(image, removal_mask, 5, cv2.INPAINT_TELEA)
    except Exception:
        plate = image.copy()

    soft = cv2.GaussianBlur(removal_mask, (9, 9), 0).astype(np.float32) / 255.0
    soft = soft[:, :, None]

    plate = plate.astype(np.float32) * soft + image.astype(np.float32) * (1.0 - soft)
    return np.clip(plate, 0, 255).astype(np.uint8)


def crop_character(image, mask, padding_ratio=0.08):
    ys, xs = np.where(mask > 20)
    if len(xs) == 0:
        return None, None, None, None

    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    width, height = x2 - x1, y2 - y1

    px = max(4, int(width * padding_ratio))
    py = max(4, int(height * padding_ratio))

    x1 = max(0, x1 - px)
    y1 = max(0, y1 - py)
    x2 = min(image.shape[1], x2 + px)
    y2 = min(image.shape[0], y2 + py)

    crop = image[y1:y2, x1:x2].copy()
    alpha = mask[y1:y2, x1:x2].copy()
    center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
    return crop, alpha, center, (x1, y1, x2, y2)


# ============================================================
# ACCURATE COLOR SEPARATION (PREVENTS BACKGROUND SELECTION)
# ============================================================

def get_color_name(rgb):
    r, g, b = [int(x) for x in rgb]
    pixel = np.uint8([[[b, g, r]]])
    hsv = cv2.cvtColor(pixel, cv2.COLOR_BGR2HSV)[0][0]
    hue, sat, val = int(hsv[0]), int(hsv[1]), int(hsv[2])

    if sat < 35:
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

        if "Paper" in c_name:
            continue

        label = f"{c_name} ({hex_code}) — {counts[idx]/total*100:.1f}%"
        detected.append({
            "label": label,
            "rgb": (r, g, b),
            "bgr": (b, g, r),
            "hex": hex_code,
            "coverage": counts[idx] / total,
        })

    if not detected:
        for idx, center in enumerate(centers):
            b, g, r = [int(x) for x in center]
            hex_code = f"#{r:02x}{g:02x}{b:02x}"
            detected.append({
                "label": f"Color ({hex_code})",
                "rgb": (r, g, b),
                "bgr": (b, g, r),
                "hex": hex_code,
                "coverage": counts[idx] / total,
            })

    detected.sort(key=lambda x: x["coverage"], reverse=True)
    return detected


def make_precise_color_mask(image, target_bgr, sharpness=25):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    target_pixel = np.uint8([[[target_bgr[0], target_bgr[1], target_bgr[2]]]])
    target_hsv = cv2.cvtColor(target_pixel, cv2.COLOR_BGR2HSV)[0][0]

    hue_tol = max(4, int(sharpness * 0.25))
    sat_tol = max(20, int(sharpness * 0.8))
    val_tol = max(20, int(sharpness * 0.8))

    min_sat = max(35, int(target_hsv[1]) - sat_tol)

    lower_bound = np.array([
        max(0, int(target_hsv[0]) - hue_tol),
        min_sat,
        max(20, int(target_hsv[2]) - val_tol),
    ], dtype=np.uint8)

    upper_bound = np.array([
        min(179, int(target_hsv[0]) + hue_tol),
        min(255, int(target_hsv[1]) + sat_tol),
        min(255, int(target_hsv[2]) + val_tol),
    ], dtype=np.uint8)

    mask = cv2.inRange(hsv_image, lower_bound, upper_bound)

    if np.count_nonzero(mask) > (image.shape[0] * image.shape[1] * 0.40):
        lower_bound[1] = max(60, lower_bound[1])
        mask = cv2.inRange(hsv_image, lower_bound, upper_bound)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


# ============================================================
# LAYER TRANSFORMATIONS & RENDERING ENGINE
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
    if canvas.shape[2] == 3:
        canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2BGRA)

    ch, cw = crop.shape[:2]
    x1, y1 = int(round(cx - cw / 2.0)), int(round(cy - ch / 2.0))
    x2, y2 = x1 + cw, y1 + ch

    if x2 <= 0 or y2 <= 0 or x1 >= canvas.shape[1] or y1 >= canvas.shape[0]:
        return canvas

    cx1, cy1 = max(0, x1), max(0, y1)
    cx2, cy2 = min(canvas.shape[1], x2), min(canvas.shape[0], y2)

    src_x1, src_y1 = cx1 - x1, cy1 - y1
    src_x2, src_y2 = src_x1 + (cx2 - cx1), src_y1 + (cy2 - cy1)

    c_crop = crop[src_y1:src_y2, src_x1:src_x2]
    a_crop = alpha[src_y1:src_y2, src_x1:src_x2]

    if c_crop.shape[2] == 3:
        c_crop = cv2.cvtColor(c_crop, cv2.COLOR_BGR2BGRA)
        c_crop[:, :, 3] = a_crop

    a = (a_crop.astype(np.float32) / 255.0)[:, :, None]
    bg_crop = canvas[cy1:cy2, cx1:cx2].astype(np.float32)

    result = c_crop.astype(np.float32) * a + bg_crop * (1.0 - a)
    result[:, :, 3] = np.maximum(bg_crop[:, :, 3], a_crop.astype(np.float32))

    canvas[cy1:cy2, cx1:cx2] = np.clip(result, 0, 255).astype(np.uint8)
    return canvas


def ease_in_out(t):
    t = float(np.clip(t, 0.0, 1.0))
    return 0.5 - 0.5 * math.cos(math.pi * t)


def render_sequential_frame(
    original_img, paper_bg, char_crop, alpha_crop, home_center, color_mask,
    global_t, walk_frac, bob_amt, sway_amt, cycles, color_mode, speed, strength, transparent_mode=False
):
    h, w = original_img.shape[:2]

    # PHASE 1: WALK-IN
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

    # PHASE 2: CROSS-FADE TO ORIGINAL & IN-SCENE COLOR MOTION
    else:
        local_t = (global_t - walk_frac) / max(1e-6, 1.0 - walk_frac)
        fade_alpha = ease_in_out(min(1.0, local_t * 2.5))

        if transparent_mode:
            base_canvas = np.zeros((h, w, 4), dtype=np.uint8)
            base_canvas = paste_layer(base_canvas, char_crop, alpha_crop, home_center[0], home_center[1])
        else:
            canvas_p1 = paste_layer(paper_bg.copy(), char_crop, alpha_crop, home_center[0], home_center[1])
            if canvas_p1.shape[2] == 3:
                canvas_p1 = cv2.cvtColor(canvas_p1, cv2.COLOR_BGR2BGRA)
            orig_rgba = cv2.cvtColor(original_img, cv2.COLOR_BGR2BGRA) if original_img.shape[2] == 3 else original_img.copy()

            base_canvas = np.clip(
                canvas_p1.astype(np.float32) * (1.0 - fade_alpha) + orig_rgba.astype(np.float32) * fade_alpha,
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
            scale = 1.0 + math.sin(phase_c) * (strength * 0.015)
            warped_c, warped_a = transform_layer(crop_c, crop_a, scale, scale, 0.0, pivot)
        else: # Storytelling Speech Cadence
            angle = math.sin(phase_c * 0.5) * strength
            warped_c, warped_a = transform_layer(crop_c, crop_a, 1.0, 1.0, angle, pivot)

        return paste_layer(base_canvas, warped_c, warped_a, cx, cy)


def build_gif(frames, fps):
    buffer = io.BytesIO()
    prepared = [
        Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGRA2RGBA if f.shape[2] == 4 else cv2.COLOR_BGR2RGB))
        for f in frames
    ]
    duration = max(20, int(1000 / max(1, fps)))
    prepared[0].save(
        buffer, format="GIF", save_all=True, append_images=prepared[1:], duration=duration, loop=0, disposal=2
    )
    return buffer.getvalue()


# ============================================================
# STREAMLIT UI & CONTROLS
# ============================================================

st.sidebar.header("🎬 Global Animation Controls")
fps = st.sidebar.select_slider("FPS", options=[8, 10, 12, 15, 20, 24], value=12)
duration = st.sidebar.slider("Total Sequence Duration (sec)", 4.0, 14.0, 7.0, 0.5)

st.sidebar.markdown("---")
st.sidebar.header("🚶 Gait Controls (Walk-In)")
walk_percent = st.sidebar.slider("Walk-In Duration (%)", 30, 70, 50)
bob_amount = st.sidebar.slider("Vertical Bobbing", 0.0, 10.0, 2.5)
sway_amount = st.sidebar.slider("Body Sway Angle", 0.0, 5.0, 1.2)
cycles = st.sidebar.slider("Walk Steps", 1, 10, 4)

st.sidebar.markdown("---")
st.sidebar.header("✂️ Clean Extraction")
outline_sensitivity = st.sidebar.slider("Outline sensitivity", 0, 100, 55)
detail_strength = st.sidebar.slider("Interior detail strength", 0, 100, 45)

st.sidebar.markdown("---")
st.sidebar.header("🎨 In-Scene Color Motion")
color_mode = st.sidebar.selectbox("Color Motion Style", COLOR_ANIMATION_MODES)
speed = st.sidebar.slider("Color Motion Speed", 1, 8, 4)
strength = st.sidebar.slider("Motion Intensity", 1, 20, 8)

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
        image = resize_image(raw_image)

        col_box1, col_box2 = st.columns(2)
        with col_box1:
            x_range = st.slider(f"Horizontal Bounding Box (X %) #{idx+1}", 0, 100, (15, 85), key=f"x_{idx}_{file.name}")
        with col_box2:
            y_range = st.slider(f"Vertical Bounding Box (Y %) #{idx+1}", 0, 100, (10, 90), key=f"y_{idx}_{file.name}")

        bbox_pct = [x_range[0], y_range[0], x_range[1], y_range[1]]

        mask_preview, bbox = extract_character_mask(
            image, bbox_pct, outline_sensitivity=outline_sensitivity, detail_strength=detail_strength
        )
        char_crop, alpha_crop, home_center, _ = crop_character(image, mask_preview)

        detected_colors = extract_dominant_colors(image)
        c1, c2 = st.columns(2)
        with c1:
            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Processed Artwork", width="stretch")
        with c2:
            selected_label = st.selectbox(
                f"Identified Colors to Animate #{idx+1}", [c["label"] for c in detected_colors], key=f"col_{idx}_{file.name}"
            )
            selected_color = next(c for c in detected_colors if c["label"] == selected_label)
            sharpness = st.slider(f"Color Sharpness #{idx+1}", 10, 60, 25, key=f"tol_{idx}_{file.name}")
            color_mask = make_precise_color_mask(image, selected_color["bgr"], sharpness)
            st.image(
                cv2.cvtColor(cv2.bitwise_and(image, image, mask=color_mask), cv2.COLOR_BGR2RGB),
                caption="Isolated Color Region", width="stretch"
            )

        single_click = st.button(f"✨ Process & Animate ({file.name})", key=f"btn_{idx}_{file.name}", width="stretch")

        if process_all or single_click:
            with st.spinner(f"Processing {file.name}..."):
                paper_bg = create_background_plate(image, mask_preview)

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
                    image, paper_bg, char_crop, alpha_crop, home_center, color_mask,
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
                    image, paper_bg, char_crop, alpha_crop, home_center, color_mask,
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
                st.markdown("**2. Transparent Overlay Sequence**")
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

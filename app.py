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
    page_title="Hand-Drawn Character Animator — Natural Walk",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🎨 Hand-Drawn Character Animator — Natural Walk & Scene Merge")

st.markdown(
    """
This version uses a **background plate + isolated character** workflow.

**Animation sequence**
1. The character starts outside the frame.
2. It walks slowly toward its original position.
3. It settles naturally instead of immediately switching.
4. The original artwork is gradually restored while the moving character fades out.
5. The background therefore appears to **return naturally**, instead of the whole canvas glowing/wiggling.

The important change is that the app no longer uses a solid paper-color canvas as the main scene.
"""
)

MAX_IMAGE_SIZE = 1100

MOTION_MODES = [
    "Slow Walk + Gentle Sway",
    "Slow Walk + Soft Bounce",
    "Slow Walk + Subtle Breathing",
    "Walk Only",
]


# ============================================================
# BASIC IMAGE UTILITIES
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


def enhance_color_temperature_and_warmth(
    image, temp_shift=8, saturation_boost=1.05
):
    """Very mild polish. Avoids changing the drawing too aggressively."""
    img = image.astype(np.float32)
    b, g, r = cv2.split(img)

    r *= 1.0 + temp_shift / 200.0
    b *= 1.0 - temp_shift / 300.0

    warmed = cv2.merge([b, g, r])
    warmed = np.clip(warmed, 0, 255).astype(np.uint8)

    hsv = cv2.cvtColor(warmed, cv2.COLOR_BGR2HSV).astype(np.float32)
    h, s, v = cv2.split(hsv)
    s = np.clip(s * saturation_boost, 0, 255)

    polished = cv2.merge([h, s, v]).astype(np.uint8)
    return cv2.cvtColor(polished, cv2.COLOR_HSV2BGR)


# ============================================================
# PAPER / BACKGROUND ESTIMATION
# ============================================================

def estimate_paper_color(image):
    """
    Estimate paper color from border pixels.
    Median is robust against a few marks touching the border.
    """
    h, w = image.shape[:2]
    band = max(8, min(30, int(min(h, w) * 0.025)))

    border = np.concatenate(
        [
            image[:band, :].reshape(-1, 3),
            image[-band:, :].reshape(-1, 3),
            image[:, :band].reshape(-1, 3),
            image[:, -band:].reshape(-1, 3),
        ],
        axis=0,
    )

    return np.median(border, axis=0).astype(np.float32)


def estimate_background_similarity(image):
    """
    Pixel-level similarity to the estimated paper color.

    Lower values = more different from paper.
    """
    bg = estimate_paper_color(image)

    diff = image.astype(np.float32) - bg.reshape(1, 1, 3)
    dist = np.sqrt(np.sum(diff * diff, axis=2))

    # Robust normalization.
    p95 = np.percentile(dist, 95)
    p95 = max(p95, 1.0)

    similarity = np.clip(1.0 - dist / p95, 0.0, 1.0)
    return similarity, bg


# ============================================================
# CHARACTER MASK EXTRACTION
# ============================================================

def keep_components_near_bbox(mask, bbox, min_area_ratio=0.00003):
    """
    Keep disconnected pieces inside/near the user-selected character box.

    This is deliberately NOT 'largest contour only'. A hand-drawn
    character can have disconnected legs, arms, eyes, hair, etc.
    """
    h, w = mask.shape[:2]

    x1, y1, x2, y2 = bbox
    box_w = max(1, x2 - x1)
    box_h = max(1, y2 - y1)

    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )

    cleaned = np.zeros_like(mask)
    min_area = max(4, int(h * w * min_area_ratio))

    # Slightly expanded bbox allows disconnected feet/hair/arms.
    pad_x = int(box_w * 0.12)
    pad_y = int(box_h * 0.12)

    ex1 = max(0, x1 - pad_x)
    ey1 = max(0, y1 - pad_y)
    ex2 = min(w, x2 + pad_x)
    ey2 = min(h, y2 + pad_y)

    for label in range(1, n_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area < min_area:
            continue

        cx, cy = centroids[label]

        if ex1 <= cx <= ex2 and ey1 <= cy <= ey2:
            cleaned[labels == label] = 255

    return cleaned


def extract_character_mask(
    image,
    bbox_pct,
    threshold_strength=30,
    preserve_details=True,
):
    """
    Robust extraction for hand-drawn artwork.

    Combines:
      - GrabCut
      - local darkness / color distance from paper
      - edge/ink information
      - component filtering

    Crucially, it does NOT collapse the character to one contour.
    """

    h, w = image.shape[:2]

    x1 = int(np.clip(bbox_pct[0], 0, 100) / 100.0 * w)
    y1 = int(np.clip(bbox_pct[1], 0, 100) / 100.0 * h)
    x2 = int(np.clip(bbox_pct[2], 0, 100) / 100.0 * w)
    y2 = int(np.clip(bbox_pct[3], 0, 100) / 100.0 * h)

    if x2 <= x1:
        x2 = min(w, x1 + 10)
    if y2 <= y1:
        y2 = min(h, y1 + 10)

    bbox = (x1, y1, x2, y2)

    # --------------------------------------------------------
    # 1. GrabCut
    # --------------------------------------------------------
    gc_mask = np.full((h, w), cv2.GC_BGD, np.uint8)

    # Outside bbox = definite background.
    gc_mask[y1:y2, x1:x2] = cv2.GC_PR_BGD

    # Inner area = probable foreground.
    margin_x = max(2, int((x2 - x1) * 0.08))
    margin_y = max(2, int((y2 - y1) * 0.08))

    ix1 = min(x2 - 1, x1 + margin_x)
    iy1 = min(y2 - 1, y1 + margin_y)
    ix2 = max(ix1 + 1, x2 - margin_x)
    iy2 = max(iy1 + 1, y2 - margin_y)

    gc_mask[iy1:iy2, ix1:ix2] = cv2.GC_PR_FGD

    bg_model = np.zeros((1, 65), np.float64)
    fg_model = np.zeros((1, 65), np.float64)

    try:
        cv2.grabCut(
            image,
            gc_mask,
            None,
            bg_model,
            fg_model,
            6,
            cv2.GC_INIT_WITH_MASK,
        )

        grab = np.where(
            (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD),
            255,
            0,
        ).astype(np.uint8)
    except Exception:
        grab = np.zeros((h, w), np.uint8)

    # --------------------------------------------------------
    # 2. Paper-distance / ink mask
    # --------------------------------------------------------
    similarity, paper_color = estimate_background_similarity(image)

    # Difference from paper. Stronger threshold means fewer pixels.
    difference = (1.0 - similarity) * 255.0

    # Adaptive threshold based on the selected strength.
    percentile = np.clip(92.0 - threshold_strength * 0.45, 72.0, 92.0)
    diff_threshold = np.percentile(
        difference[y1:y2, x1:x2],
        percentile,
    )

    paper_mask = np.zeros((h, w), np.uint8)
    paper_mask[
        (difference >= diff_threshold)
    ] = 255

    # Restrict paper-derived mask to the selected region.
    region = np.zeros((h, w), np.uint8)
    region[y1:y2, x1:x2] = 255
    paper_mask = cv2.bitwise_and(paper_mask, region)

    # --------------------------------------------------------
    # 3. Ink / line structure mask
    # --------------------------------------------------------
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    crop_gray = gray[y1:y2, x1:x2]

    # Local background normalization helps with uneven paper lighting.
    blur_size = max(15, int(min(crop_gray.shape[:2]) * 0.08))
    if blur_size % 2 == 0:
        blur_size += 1
    blur_size = min(101, blur_size)

    local_bg = cv2.GaussianBlur(crop_gray, (blur_size, blur_size), 0)

    local_darkness = cv2.subtract(local_bg, crop_gray)

    dark_threshold = np.percentile(
        local_darkness,
        np.clip(93 - threshold_strength * 0.35, 70, 93),
    )

    ink_crop = np.where(local_darkness >= dark_threshold, 255, 0).astype(
        np.uint8
    )

    # Also preserve genuinely dark colored strokes.
    hsv_crop = cv2.cvtColor(image[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)
    saturation = hsv_crop[:, :, 1]
    value = hsv_crop[:, :, 2]

    color_ink = (
        (saturation > max(35, 65 - threshold_strength))
        & (value < 245)
    ).astype(np.uint8) * 255

    ink_crop = cv2.bitwise_or(ink_crop, color_ink)

    ink_mask = np.zeros((h, w), np.uint8)
    ink_mask[y1:y2, x1:x2] = ink_crop

    # --------------------------------------------------------
    # 4. Combine
    # --------------------------------------------------------
    if preserve_details:
        combined = cv2.bitwise_or(grab, paper_mask)
        combined = cv2.bitwise_or(combined, ink_mask)
    else:
        combined = cv2.bitwise_or(grab, paper_mask)

    # Never allow anything outside bbox.
    combined = cv2.bitwise_and(combined, region)

    # Close tiny gaps but do not fill huge holes.
    k_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    combined = cv2.morphologyEx(
        combined, cv2.MORPH_CLOSE, k_small, iterations=1
    )
    combined = cv2.morphologyEx(
        combined, cv2.MORPH_OPEN, k_small, iterations=1
    )

    # --------------------------------------------------------
    # 5. Keep relevant disconnected components
    # --------------------------------------------------------
    combined = keep_components_near_bbox(combined, bbox)

    # If component filtering was too aggressive, fall back to bbox mask.
    area = np.count_nonzero(combined)
    bbox_area = max(1, (x2 - x1) * (y2 - y1))

    if area < bbox_area * 0.002:
        combined = cv2.bitwise_or(grab, ink_mask)
        combined = cv2.bitwise_and(combined, region)

    # Small feather only. This prevents harsh cutout edges.
    combined = cv2.GaussianBlur(combined, (3, 3), 0)

    return combined, bbox, paper_color


# ============================================================
# BACKGROUND PLATE
# ============================================================

def create_background_plate(image, character_mask):
    """
    Remove the extracted character from the original image using
    inpainting. This gives the walk-in animation a real background
    rather than a flat paper-color background.
    """

    h, w = character_mask.shape[:2]

    # Slight expansion is important: otherwise a halo of character
    # pixels remains behind the walking character.
    kernel_size = max(5, int(min(h, w) * 0.012))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel_size = min(31, kernel_size)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_size, kernel_size),
    )

    removal_mask = cv2.dilate(character_mask, kernel, iterations=1)

    # Inpainting works better with a hard mask.
    hard_mask = np.where(removal_mask > 25, 255, 0).astype(np.uint8)

    # Two passes: Telea first, then mild smoothing.
    try:
        plate = cv2.inpaint(
            image,
            hard_mask,
            5,
            cv2.INPAINT_TELEA,
        )
    except Exception:
        plate = image.copy()

    # Keep original background untouched outside the removal zone.
    # This minimizes unintended image changes.
    soft = cv2.GaussianBlur(hard_mask, (9, 9), 0).astype(np.float32) / 255.0
    soft = soft[:, :, None]

    plate = (
        plate.astype(np.float32) * soft
        + image.astype(np.float32) * (1.0 - soft)
    )

    return np.clip(plate, 0, 255).astype(np.uint8)


# ============================================================
# CHARACTER CROP
# ============================================================

def crop_character(image, mask, padding_ratio=0.10):
    ys, xs = np.where(mask > 20)

    if len(xs) == 0:
        return None, None, None, None

    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1

    width = x2 - x1
    height = y2 - y1

    px = max(4, int(width * padding_ratio))
    py = max(4, int(height * padding_ratio))

    x1 = max(0, x1 - px)
    y1 = max(0, y1 - py)
    x2 = min(image.shape[1], x2 + px)
    y2 = min(image.shape[0], y2 + py)

    crop = image[y1:y2, x1:x2].copy()
    alpha = mask[y1:y2, x1:x2].copy()

    center = (
        (x1 + x2) / 2.0,
        (y1 + y2) / 2.0,
    )

    return crop, alpha, center, (x1, y1, x2, y2)


# ============================================================
# ALPHA COMPOSITING
# ============================================================

def paste_rgba_like(canvas, crop, alpha, center_x, center_y):
    """
    High-quality alpha compositing.
    """
    if canvas.ndim != 3:
        return canvas

    h, w = canvas.shape[:2]
    ch, cw = crop.shape[:2]

    x1 = int(round(center_x - cw / 2))
    y1 = int(round(center_y - ch / 2))
    x2 = x1 + cw
    y2 = y1 + ch

    if x2 <= 0 or y2 <= 0 or x1 >= w or y1 >= h:
        return canvas

    cx1 = max(0, x1)
    cy1 = max(0, y1)
    cx2 = min(w, x2)
    cy2 = min(h, y2)

    sx1 = cx1 - x1
    sy1 = cy1 - y1
    sx2 = sx1 + (cx2 - cx1)
    sy2 = sy1 + (cy2 - cy1)

    src = crop[sy1:sy2, sx1:sx2].astype(np.float32)
    a = alpha[sy1:sy2, sx1:sx2].astype(np.float32) / 255.0
    a = a[:, :, None]

    dst = canvas[cy1:cy2, cx1:cx2].astype(np.float32)

    result = src * a + dst * (1.0 - a)

    canvas[cy1:cy2, cx1:cx2] = np.clip(result, 0, 255).astype(np.uint8)
    return canvas


def transform_character(
    crop,
    alpha,
    scale_x=1.0,
    scale_y=1.0,
    angle=0.0,
):
    h, w = crop.shape[:2]

    new_w = max(2, int(w * scale_x))
    new_h = max(2, int(h * scale_y))

    resized = cv2.resize(
        crop,
        (new_w, new_h),
        interpolation=cv2.INTER_CUBIC,
    )

    resized_alpha = cv2.resize(
        alpha,
        (new_w, new_h),
        interpolation=cv2.INTER_CUBIC,
    )

    center = (new_w / 2.0, new_h / 2.0)

    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    cos = abs(M[0, 0])
    sin = abs(M[0, 1])

    bw = max(2, int(new_h * sin + new_w * cos))
    bh = max(2, int(new_h * cos + new_w * sin))

    M[0, 2] += bw / 2.0 - center[0]
    M[1, 2] += bh / 2.0 - center[1]

    warped = cv2.warpAffine(
        resized,
        M,
        (bw, bh),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )

    warped_alpha = cv2.warpAffine(
        resized_alpha,
        M,
        (bw, bh),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    return warped, warped_alpha


# ============================================================
# EASING
# ============================================================

def ease_in_out(t):
    t = float(np.clip(t, 0.0, 1.0))
    return 0.5 - 0.5 * math.cos(math.pi * t)


def smoothstep(t):
    t = float(np.clip(t, 0.0, 1.0))
    return t * t * (3.0 - 2.0 * t)


def lerp(a, b, t):
    return a + (b - a) * t


# ============================================================
# WALK CYCLE
# ============================================================

def walking_pose(
    local_t,
    mode,
    step_count,
    bob_amount,
    sway_amount,
):
    """
    Subtle 2D motion only.

    We deliberately keep the amplitude low because large transforms
    make a hand-drawn cutout look like a sticker sliding around.
    """

    phase = local_t * step_count * math.pi * 2.0

    if mode == "Walk Only":
        bob = 0.0
        sway = 0.0
        scale_y = 1.0

    elif mode == "Slow Walk + Soft Bounce":
        bob = math.sin(phase) * bob_amount
        sway = math.sin(phase + math.pi / 2.0) * sway_amount
        scale_y = 1.0 + math.sin(phase) * 0.006

    elif mode == "Slow Walk + Subtle Breathing":
        bob = math.sin(phase) * bob_amount * 0.65
        sway = math.sin(phase * 0.5) * sway_amount
        scale_y = 1.0 + math.sin(phase * 0.5) * 0.012

    else:
        bob = math.sin(phase) * bob_amount
        sway = math.sin(phase + math.pi / 2.0) * sway_amount
        scale_y = 1.0 + math.sin(phase) * 0.004

    return bob, sway, 1.0 / scale_y, scale_y


# ============================================================
# MAIN FRAME RENDER
# ============================================================

def render_frame(
    original,
    background_plate,
    char_crop,
    char_alpha,
    home_center,
    t,
    walk_fraction,
    settle_fraction,
    mode,
    step_count,
    bob_amount,
    sway_amount,
    entrance_side,
    merge_strength,
):
    """
    Timeline:

      [ WALK ] -> [ SETTLE ] -> [ MERGE TO ORIGINAL ]

    During MERGE:
      background plate fades into original artwork
      while moving character fades out.

    This avoids the previous 'paper background -> original background'
    jump.
    """

    h, w = original.shape[:2]

    # --------------------------------------------------------
    # Timeline
    # --------------------------------------------------------
    walk_end = walk_fraction
    settle_end = min(
        0.95,
        walk_end + settle_fraction,
    )

    # Start from outside the canvas.
    char_w = char_crop.shape[1]

    if entrance_side == "Left":
        start_x = -char_w * 0.75
    else:
        start_x = w + char_w * 0.75

    target_x = home_center[0]
    target_y = home_center[1]

    # --------------------------------------------------------
    # Phase 1: WALK IN
    # --------------------------------------------------------
    if t < walk_end:
        p = ease_in_out(t / max(1e-6, walk_end))

        # Slightly slow at the end of the walk.
        p = smoothstep(p)

        current_x = lerp(start_x, target_x, p)

        # Character should settle around the correct height.
        current_y = target_y

        bob, sway, sx, sy = walking_pose(
            p,
            mode,
            step_count,
            bob_amount,
            sway_amount,
        )

        current_y += bob

        warped_c, warped_a = transform_character(
            char_crop,
            char_alpha,
            sx,
            sy,
            sway,
        )

        frame = background_plate.copy()

        return paste_rgba_like(
            frame,
            warped_c,
            warped_a,
            current_x,
            current_y,
        )

    # --------------------------------------------------------
    # Phase 2: SETTLE
    # --------------------------------------------------------
    if t < settle_end:
        p = smoothstep(
            (t - walk_end) / max(1e-6, settle_end - walk_end)
        )

        # Motion decreases to zero.
        bob, sway, sx, sy = walking_pose(
            1.0 - p,
            mode,
            step_count,
            bob_amount * (1.0 - p),
            sway_amount * (1.0 - p),
        )

        current_x = target_x
        current_y = target_y + bob

        warped_c, warped_a = transform_character(
            char_crop,
            char_alpha,
            sx,
            sy,
            sway,
        )

        frame = background_plate.copy()

        return paste_rgba_like(
            frame,
            warped_c,
            warped_a,
            current_x,
            current_y,
        )

    # --------------------------------------------------------
    # Phase 3: MERGE TO ORIGINAL
    # --------------------------------------------------------
    p = smoothstep(
        (t - settle_end) / max(1e-6, 1.0 - settle_end)
    )

    # First half: almost completely settled.
    # Second half: stronger restoration.
    merge = smoothstep(np.clip(p * 1.20, 0.0, 1.0))
    merge = merge ** max(0.35, 1.0 / max(0.1, merge_strength))

    moving = background_plate.copy()

    # Character remains visible at first and then gently disappears.
    character_alpha_factor = 1.0 - merge

    warped_c, warped_a = transform_character(
        char_crop,
        char_alpha,
        1.0,
        1.0,
        0.0,
    )

    warped_a = np.clip(
        warped_a.astype(np.float32) * character_alpha_factor,
        0,
        255,
    ).astype(np.uint8)

    moving = paste_rgba_like(
        moving,
        warped_c,
        warped_a,
        target_x,
        target_y,
    )

    # Restore the original image gradually.
    frame = (
        moving.astype(np.float32) * (1.0 - merge)
        + original.astype(np.float32) * merge
    )

    return np.clip(frame, 0, 255).astype(np.uint8)


# ============================================================
# GIF / APNG EXPORT
# ============================================================

def build_gif(frames, fps):
    buffer = io.BytesIO()

    prepared = [
        Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        for frame in frames
    ]

    duration = max(20, int(1000 / max(1, fps)))

    prepared[0].save(
        buffer,
        format="GIF",
        save_all=True,
        append_images=prepared[1:],
        duration=duration,
        loop=0,
        disposal=2,
    )

    return buffer.getvalue()


def build_apng(frames, fps):
    """
    APNG preserves full RGB quality better than GIF.
    """
    buffer = io.BytesIO()

    prepared = [
        Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert("RGB")
        for frame in frames
    ]

    duration = max(20, int(1000 / max(1, fps)))

    prepared[0].save(
        buffer,
        format="PNG",
        save_all=True,
        append_images=prepared[1:],
        duration=duration,
        loop=0,
        disposal=2,
    )

    return buffer.getvalue()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("🎬 Timing")

fps = st.sidebar.select_slider(
    "FPS",
    options=[8, 10, 12, 15, 20, 24],
    value=12,
)

duration = st.sidebar.slider(
    "Total Duration (seconds)",
    5.0,
    16.0,
    9.0,
    0.5,
)

st.sidebar.markdown("---")

st.sidebar.header("🚶 Walk")

walk_percent = st.sidebar.slider(
    "Walking time (%)",
    25,
    60,
    42,
)

settle_percent = st.sidebar.slider(
    "Settle time (%)",
    5,
    25,
    12,
)

entrance_side = st.sidebar.selectbox(
    "Entrance side",
    ["Left", "Right"],
)

step_count = st.sidebar.slider(
    "Slow walking steps",
    2,
    12,
    5,
)

bob_amount = st.sidebar.slider(
    "Walking vertical motion",
    0.0,
    12.0,
    3.0,
    0.5,
)

sway_amount = st.sidebar.slider(
    "Walking body sway",
    0.0,
    5.0,
    1.5,
    0.25,
)

motion_mode = st.sidebar.selectbox(
    "Motion style",
    MOTION_MODES,
)

st.sidebar.markdown("---")

st.sidebar.header("🧩 Character Extraction")

threshold_strength = st.sidebar.slider(
    "Character extraction strength",
    10,
    70,
    35,
)

preserve_details = st.sidebar.checkbox(
    "Preserve disconnected details",
    value=True,
)

padding = st.sidebar.slider(
    "Character edge padding",
    2,
    20,
    10,
)

st.sidebar.markdown("---")

st.sidebar.header("🌅 Scene Merge")

merge_strength = st.sidebar.slider(
    "Merge smoothness",
    1,
    10,
    5,
)

st.sidebar.markdown("---")

st.sidebar.header("✨ Very Mild Color Polish")

enable_enhancement = st.sidebar.checkbox(
    "Enable",
    value=False,
)

temp_shift = st.sidebar.slider(
    "Warmth",
    0,
    20,
    6,
)


# ============================================================
# UPLOAD
# ============================================================

uploaded_files = st.file_uploader(
    "Upload one or more drawings",
    type=["jpg", "jpeg", "png", "webp"],
    accept_multiple_files=True,
)

if not uploaded_files:
    st.info(
        "Upload your drawing above. For the best result, set the bounding "
        "box around the entire character, including disconnected arms, "
        "legs, hair, and small details."
    )
    st.stop()


# ============================================================
# PROCESS FILES
# ============================================================

zip_export_files = {}

for idx, uploaded in enumerate(uploaded_files):

    st.markdown("---")
    st.subheader(f"🖼️ Drawing {idx + 1}: {uploaded.name}")

    uploaded.seek(0)
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)

    raw = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if raw is None:
        st.error(f"Could not read {uploaded.name}.")
        continue

    raw = auto_rotate_vertical(raw)

    if enable_enhancement:
        image = enhance_color_temperature_and_warmth(
            raw,
            temp_shift=temp_shift,
            saturation_boost=1.03,
        )
    else:
        image = raw.copy()

    image = resize_image(image)

    h, w = image.shape[:2]

    # --------------------------------------------------------
    # BBOX
    # --------------------------------------------------------
    cbox1, cbox2 = st.columns(2)

    with cbox1:
        x_range = st.slider(
            f"Character X range #{idx + 1}",
            0,
            100,
            (15, 85),
            key=f"x_{idx}_{uploaded.name}",
        )

    with cbox2:
        y_range = st.slider(
            f"Character Y range #{idx + 1}",
            0,
            100,
            (10, 90),
            key=f"y_{idx}_{uploaded.name}",
        )

    bbox_pct = [
        x_range[0],
        y_range[0],
        x_range[1],
        y_range[1],
    ]

    # --------------------------------------------------------
    # PREVIEW EXTRACTION
    # --------------------------------------------------------
    mask_preview, bbox, paper_color = extract_character_mask(
        image,
        bbox_pct,
        threshold_strength=threshold_strength,
        preserve_details=preserve_details,
    )

    char_preview, alpha_preview, home_preview, char_bbox = crop_character(
        image,
        mask_preview,
        padding_ratio=padding / 100.0,
    )

    preview1, preview2, preview3 = st.columns(3)

    with preview1:
        st.image(
            cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
            caption="Original artwork",
            width="stretch",
        )

    with preview2:
        mask_visual = cv2.cvtColor(mask_preview, cv2.COLOR_GRAY2RGB)

        # Red overlay makes accidental background selection obvious.
        overlay = image.copy()
        red = np.zeros_like(image)
        red[:, :, 2] = 255

        m = mask_preview.astype(np.float32) / 255.0
        m = m[:, :, None]

        overlay = (
            overlay.astype(np.float32) * (1 - 0.38 * m)
            + red.astype(np.float32) * (0.38 * m)
        )

        overlay = np.clip(overlay, 0, 255).astype(np.uint8)

        st.image(
            cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB),
            caption="Character extraction preview (red = selected)",
            width="stretch",
        )

    with preview3:
        if char_preview is not None:
            isolated = np.ones_like(char_preview) * 255
            a = alpha_preview.astype(np.float32) / 255.0
            a = a[:, :, None]

            isolated = (
                char_preview.astype(np.float32) * a
                + isolated.astype(np.float32) * (1 - a)
            )

            isolated = np.clip(isolated, 0, 255).astype(np.uint8)

            st.image(
                cv2.cvtColor(isolated, cv2.COLOR_BGR2RGB),
                caption="Isolated character",
                width="stretch",
            )
        else:
            st.error("No character pixels detected.")

    # --------------------------------------------------------
    # PROCESS BUTTON
    # --------------------------------------------------------
    process = st.button(
        f"🎬 Animate {uploaded.name}",
        key=f"process_{idx}_{uploaded.name}",
        type="primary",
        width="stretch",
    )

    if not process:
        continue

    if char_preview is None:
        st.error(
            "The character could not be isolated. Expand the bounding box "
            "or increase Character Extraction Strength."
        )
        continue

    with st.spinner("Building clean background plate..."):
        background_plate = create_background_plate(
            image,
            mask_preview,
        )

    # --------------------------------------------------------
    # SHOW BACKGROUND PLATE
    # --------------------------------------------------------
    st.image(
        cv2.cvtColor(background_plate, cv2.COLOR_BGR2RGB),
        caption="Background plate — character removed",
        width="stretch",
    )

    # --------------------------------------------------------
    # FINAL CHARACTER CROP
    # --------------------------------------------------------
    char_crop, char_alpha, home_center, char_bbox = crop_character(
        image,
        mask_preview,
        padding_ratio=padding / 100.0,
    )

    if char_crop is None:
        st.error("Character crop failed.")
        continue

    frame_count = max(
        12,
        int(round(fps * duration)),
    )

    walk_fraction = walk_percent / 100.0
    settle_fraction = settle_percent / 100.0

    progress = st.progress(
        0,
        text="Rendering natural walk...",
    )

    frames = []

    for i in range(frame_count):
        t = i / max(1, frame_count - 1)

        frame = render_frame(
            original=image,
            background_plate=background_plate,
            char_crop=char_crop,
            char_alpha=char_alpha,
            home_center=home_center,
            t=t,
            walk_fraction=walk_fraction,
            settle_fraction=settle_fraction,
            mode=motion_mode,
            step_count=step_count,
            bob_amount=bob_amount,
            sway_amount=sway_amount,
            entrance_side=entrance_side,
            merge_strength=merge_strength,
        )

        frames.append(frame)
        progress.progress(
            (i + 1) / frame_count,
            text=f"Rendering frame {i + 1}/{frame_count}",
        )

    progress.empty()

    # --------------------------------------------------------
    # EXPORT
    # --------------------------------------------------------
    gif_data = build_gif(frames, fps)

    # APNG may be larger but keeps the drawing cleaner.
    apng_data = build_apng(frames, fps)

    gif_name = f"natural_walk_{os.path.splitext(uploaded.name)[0]}.gif"
    apng_name = f"natural_walk_{os.path.splitext(uploaded.name)[0]}.png"

    zip_export_files[gif_name] = gif_data
    zip_export_files[apng_name] = apng_data

    st.markdown("### 🎬 Result")

    st.image(
        gif_data,
        caption="Natural walk → settle → merge back into original scene",
        width="stretch",
    )

    d1, d2 = st.columns(2)

    with d1:
        st.download_button(
            "⬇️ Download GIF",
            data=gif_data,
            file_name=gif_name,
            mime="image/gif",
            key=f"gif_{idx}_{uploaded.name}",
            width="stretch",
        )

    with d2:
        st.download_button(
            "⬇️ Download High-Quality APNG",
            data=apng_data,
            file_name=apng_name,
            mime="image/png",
            key=f"apng_{idx}_{uploaded.name}",
            width="stretch",
        )


# ============================================================
# ZIP EXPORT
# ============================================================

if zip_export_files:
    st.markdown("---")
    st.subheader("📦 Bulk Download")

    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(
        zip_buffer,
        "w",
        zipfile.ZIP_DEFLATED,
    ) as z:
        for name, data in zip_export_files.items():
            z.writestr(name, data)

    st.download_button(
        "📦 Download All Animations",
        data=zip_buffer.getvalue(),
        file_name="natural_walk_animations.zip",
        mime="application/zip",
        type="primary",
        width="stretch",
    )


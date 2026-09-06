
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
    page_title="Hand-Drawn Character Animator — Clean Extraction",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🎨 Hand-Drawn Character Animator — Clean Character Extraction")

st.markdown(
    """
### New extraction system
This version is designed for **photographed hand-drawn artwork on paper**.

Instead of treating every darker/pinker paper pixel as foreground, it uses:

- the **dark hand-drawn outline** as the main character boundary,
- closed-contour filling to recover the character's interior,
- local colour/ink information **only inside the character silhouette**,
- component filtering to reject hearts, stars and other decorations,
- a real background plate made from the original scene,
- slow **walk → settle → merge back into the original artwork** animation.

The extraction preview is the most important part: **red should cover the llama, not the surrounding paper.**
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
# IMAGE UTILITIES
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
        (
            max(2, int(w * scale)),
            max(2, int(h * scale)),
        ),
        interpolation=cv2.INTER_AREA,
    )


def enhance_color_temperature_and_warmth(
    image,
    temp_shift=6,
    saturation_boost=1.03,
):
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
# PAPER ESTIMATION
# ============================================================

def estimate_paper_color(image):
    """
    Estimate the paper colour from the outer border.

    The result is used only as a weak reference. It is NOT used
    to globally classify the whole photograph as foreground.
    """

    h, w = image.shape[:2]

    band = max(
        5,
        min(
            25,
            int(min(h, w) * 0.025),
        ),
    )

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


def paper_distance(image):
    bg = estimate_paper_color(image)

    diff = image.astype(np.float32) - bg.reshape(1, 1, 3)

    return np.sqrt(
        np.sum(diff * diff, axis=2)
    )


# ============================================================
# MASK HELPERS
# ============================================================

def bbox_from_percentages(image, bbox_pct):
    h, w = image.shape[:2]

    x1 = int(
        np.clip(
            bbox_pct[0],
            0,
            100,
        )
        * w
        / 100.0
    )

    y1 = int(
        np.clip(
            bbox_pct[1],
            0,
            100,
        )
        * h
        / 100.0
    )

    x2 = int(
        np.clip(
            bbox_pct[2],
            0,
            100,
        )
        * w
        / 100.0
    )

    y2 = int(
        np.clip(
            bbox_pct[3],
            0,
            100,
        )
        * h
        / 100.0
    )

    x1, x2 = sorted([x1, x2])
    y1, y2 = sorted([y1, y2])

    x2 = max(x2, x1 + 5)
    y2 = max(y2, y1 + 5)

    x2 = min(x2, w)
    y2 = min(y2, h)

    return x1, y1, x2, y2


def largest_component(mask):
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8,
    )

    if n <= 1:
        return np.zeros_like(mask)

    areas = stats[1:, cv2.CC_STAT_AREA]

    index = 1 + int(np.argmax(areas))

    return np.where(
        labels == index,
        255,
        0,
    ).astype(np.uint8)


def remove_small_components(
    mask,
    minimum_area,
):
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8,
    )

    cleaned = np.zeros_like(mask)

    for label in range(1, n):
        area = stats[label, cv2.CC_STAT_AREA]

        if area >= minimum_area:
            cleaned[labels == label] = 255

    return cleaned


# ============================================================
# OUTLINE DETECTION
# ============================================================

def detect_dark_outline(
    crop,
    sensitivity=55,
):
    """
    Detect the hand-drawn dark outline.

    This deliberately favours dark/local-contrast strokes instead
    of global colour difference. This is the key change from the
    previous algorithm.
    """

    gray = cv2.cvtColor(
        crop,
        cv2.COLOR_BGR2GRAY,
    )

    h, w = gray.shape

    # --------------------------------------------------------
    # Local darkness
    # --------------------------------------------------------

    blur_size = int(
        np.clip(
            min(h, w) * 0.07,
            15,
            71,
        )
    )

    if blur_size % 2 == 0:
        blur_size += 1

    local_bg = cv2.GaussianBlur(
        gray,
        (blur_size, blur_size),
        0,
    )

    darkness = cv2.subtract(
        local_bg,
        gray,
    )

    # Higher sensitivity = slightly easier detection.
    darkness_percentile = np.interp(
        sensitivity,
        [0, 100],
        [97, 82],
    )

    threshold_dark = np.percentile(
        darkness,
        darkness_percentile,
    )

    local_mask = np.where(
        darkness >= threshold_dark,
        255,
        0,
    ).astype(np.uint8)

    # --------------------------------------------------------
    # Absolute dark ink
    # --------------------------------------------------------

    hsv = cv2.cvtColor(
        crop,
        cv2.COLOR_BGR2HSV,
    )

    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    absolute_value_threshold = int(
        np.interp(
            sensitivity,
            [0, 100],
            [105, 170],
        )
    )

    absolute_dark = (
        (value <= absolute_value_threshold)
        &
        (
            (saturation >= 10)
            |
            (value <= 125)
        )
    )

    absolute_dark = (
        absolute_dark.astype(np.uint8)
        * 255
    )

    ink = cv2.bitwise_or(
        local_mask,
        absolute_dark,
    )

    # --------------------------------------------------------
    # Remove tiny paper texture
    # --------------------------------------------------------

    open_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (3, 3),
    )

    ink = cv2.morphologyEx(
        ink,
        cv2.MORPH_OPEN,
        open_kernel,
        iterations=1,
    )

    # --------------------------------------------------------
    # Close small breaks in hand-drawn outline
    # --------------------------------------------------------

    close_size = int(
        np.interp(
            sensitivity,
            [0, 100],
            [5, 11],
        )
    )

    if close_size % 2 == 0:
        close_size += 1

    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (
            close_size,
            close_size,
        ),
    )

    ink = cv2.morphologyEx(
        ink,
        cv2.MORPH_CLOSE,
        close_kernel,
        iterations=1,
    )

    return ink


# ============================================================
# CLOSED OUTLINE -> CHARACTER SILHOUETTE
# ============================================================

def contour_fill_from_outline(
    outline,
    sensitivity=55,
):
    """
    Convert dark outline into a filled character silhouette.

    Decorations such as hearts are usually separate contours.
    The character contour is selected using a score based on:

      - contour area,
      - position inside the bbox,
      - shape size,
      - proximity to the crop centre.

    This is much safer than globally thresholding paper colour.
    """

    h, w = outline.shape[:2]

    contours, hierarchy = cv2.findContours(
        outline,
        cv2.RETR_CCOMP,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if not contours:
        return np.zeros_like(outline)

    crop_area = float(h * w)
    cx0 = w / 2.0
    cy0 = h / 2.0

    candidates = []

    for i, contour in enumerate(contours):

        area = cv2.contourArea(contour)

        if area < crop_area * 0.003:
            continue

        x, y, cw, ch = cv2.boundingRect(
            contour
        )

        bbox_area = float(
            max(1, cw * ch)
        )

        fill_ratio = area / bbox_area

        center_x = x + cw / 2.0
        center_y = y + ch / 2.0

        distance = math.sqrt(
            (
                (center_x - cx0)
                / max(1, w)
            )
            ** 2
            +
            (
                (center_y - cy0)
                / max(1, h)
            )
            ** 2
        )

        centre_score = max(
            0.0,
            1.0 - distance * 2.5,
        )

        size_score = min(
            1.0,
            area / (crop_area * 0.30),
        )

        # Large, central contours win.
        score = (
            area
            * (
                0.55
                + 0.30 * centre_score
                + 0.15 * size_score
            )
        )

        candidates.append(
            (
                score,
                i,
                area,
                fill_ratio,
            )
        )

    if not candidates:
        return np.zeros_like(outline)

    candidates.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    # --------------------------------------------------------
    # Select the best contour.
    # --------------------------------------------------------

    best_index = candidates[0][1]

    silhouette = np.zeros_like(
        outline
    )

    cv2.drawContours(
        silhouette,
        contours,
        best_index,
        255,
        thickness=cv2.FILLED,
    )

    # --------------------------------------------------------
    # Preserve holes if they are real contour children.
    # --------------------------------------------------------

    if hierarchy is not None:
        children = []

        for i, c in enumerate(
            hierarchy[0]
        ):
            parent = c[3]

            if parent == best_index:
                children.append(i)

        for child_index in children:
            child_area = cv2.contourArea(
                contours[child_index]
            )

            # Only remove reasonably large enclosed holes.
            if child_area > crop_area * 0.0004:
                cv2.drawContours(
                    silhouette,
                    contours,
                    child_index,
                    0,
                    thickness=cv2.FILLED,
                )

    # --------------------------------------------------------
    # If the outline is fragmented, gently connect nearby
    # character pieces. Do not aggressively flood the bbox.
    # --------------------------------------------------------

    bridge_size = int(
        np.interp(
            sensitivity,
            [0, 100],
            [3, 7],
        )
    )

    if bridge_size % 2 == 0:
        bridge_size += 1

    bridge_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (
            bridge_size,
            bridge_size,
        ),
    )

    # Only a very small closing here.
    silhouette = cv2.morphologyEx(
        silhouette,
        cv2.MORPH_CLOSE,
        bridge_kernel,
        iterations=1,
    )

    return silhouette


# ============================================================
# CHARACTER EXTRACTION
# ============================================================

def extract_character_mask(
    image,
    bbox_pct,
    outline_sensitivity=55,
    detail_strength=45,
):
    """
    Main extraction algorithm.

    IMPORTANT:
    The old algorithm used paper-distance pixels as foreground
    across the entire bbox. That is exactly what caused the large
    red paper patches in the user's screenshot.

    This version does the opposite:

        1. find the dark character outline,
        2. build a silhouette from it,
        3. use colour only INSIDE that silhouette,
        4. attach nearby disconnected character details,
        5. reject surrounding decorations.
    """

    x1, y1, x2, y2 = bbox_from_percentages(
        image,
        bbox_pct,
    )

    crop = image[
        y1:y2,
        x1:x2,
    ].copy()

    ch, cw = crop.shape[:2]

    if ch < 10 or cw < 10:
        return (
            np.zeros(
                image.shape[:2],
                dtype=np.uint8,
            ),
            (x1, y1, x2, y2),
        )

    # --------------------------------------------------------
    # STEP 1: Dark outline
    # --------------------------------------------------------

    outline = detect_dark_outline(
        crop,
        sensitivity=outline_sensitivity,
    )

    # --------------------------------------------------------
    # STEP 2: Main filled silhouette
    # --------------------------------------------------------

    silhouette = contour_fill_from_outline(
        outline,
        sensitivity=outline_sensitivity,
    )

    # --------------------------------------------------------
    # STEP 3: If contour detection fails, use GrabCut,
    # but with strict seeds rather than treating the whole bbox
    # as probable foreground.
    # --------------------------------------------------------

    silhouette_area = np.count_nonzero(
        silhouette
    )

    bbox_area = max(
        1,
        cw * ch,
    )

    if (
        silhouette_area < bbox_area * 0.01
        or silhouette_area > bbox_area * 0.88
    ):
        silhouette = fallback_grabcut_mask(
            crop,
            outline,
        )

    # --------------------------------------------------------
    # STEP 4: Recover coloured interior pixels.
    #
    # This is now restricted to the silhouette.
    # Therefore pink body pixels can be recovered without
    # selecting the grey paper elsewhere.
    # --------------------------------------------------------

    hsv = cv2.cvtColor(
        crop,
        cv2.COLOR_BGR2HSV,
    )

    saturation = hsv[:, :, 1].astype(
        np.float32
    )

    value = hsv[:, :, 2].astype(
        np.float32
    )

    # Colourful pixels inside the silhouette.
    colour_pixels = (
        (saturation > np.interp(
            detail_strength,
            [0, 100],
            [22, 10],
        ))
        &
        (value < 252)
    ).astype(np.uint8) * 255

    # Dark pixels inside silhouette.
    gray = cv2.cvtColor(
        crop,
        cv2.COLOR_BGR2GRAY,
    )

    local_bg = cv2.GaussianBlur(
        gray,
        (21, 21),
        0,
    )

    local_darkness = cv2.subtract(
        local_bg,
        gray,
    )

    dark_threshold = np.percentile(
        local_darkness,
        np.interp(
            detail_strength,
            [0, 100],
            [97, 75],
        ),
    )

    local_dark = np.where(
        local_darkness >= dark_threshold,
        255,
        0,
    ).astype(np.uint8)

    interior_details = cv2.bitwise_or(
        colour_pixels,
        local_dark,
    )

    # --------------------------------------------------------
    # STEP 5: Add details ONLY when they are close to the
    # silhouette. This prevents hearts / stars from being
    # accidentally selected.
    # --------------------------------------------------------

    proximity_size = int(
        np.clip(
            min(ch, cw) * 0.055,
            9,
            31,
        )
    )

    if proximity_size % 2 == 0:
        proximity_size += 1

    proximity_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (
            proximity_size,
            proximity_size,
        ),
    )

    silhouette_neighbourhood = cv2.dilate(
        silhouette,
        proximity_kernel,
        iterations=1,
    )

    nearby_details = cv2.bitwise_and(
        interior_details,
        silhouette_neighbourhood,
    )

    # The silhouette itself remains dominant.
    combined = cv2.bitwise_or(
        silhouette,
        nearby_details,
    )

    # --------------------------------------------------------
    # STEP 6: Remove tiny isolated pieces.
    # --------------------------------------------------------

    min_component = max(
        8,
        int(bbox_area * 0.000035),
    )

    combined = remove_small_components(
        combined,
        min_component,
    )

    # --------------------------------------------------------
    # STEP 7: Never allow pixels outside the user bbox.
    # --------------------------------------------------------

    combined = np.where(
        combined > 0,
        255,
        0,
    ).astype(np.uint8)

    full_mask = np.zeros(
        image.shape[:2],
        dtype=np.uint8,
    )

    full_mask[
        y1:y2,
        x1:x2,
    ] = combined

    # --------------------------------------------------------
    # STEP 8: Very small feather.
    # --------------------------------------------------------

    full_mask = cv2.GaussianBlur(
        full_mask,
        (3, 3),
        0,
    )

    return (
        full_mask,
        (x1, y1, x2, y2),
    )


# ============================================================
# GRABCUT FALLBACK
# ============================================================

def fallback_grabcut(
    crop,
    outline,
):
    h, w = crop.shape[:2]

    mask = np.full(
        (h, w),
        cv2.GC_PR_BGD,
        dtype=np.uint8,
    )

    # Border is definite background.
    border = max(
        2,
        int(min(h, w) * 0.025),
    )

    mask[
        :border,
        :,
    ] = cv2.GC_BGD

    mask[
        -border:,
        :,
    ] = cv2.GC_BGD

    mask[
        :,
        :border,
    ] = cv2.GC_BGD

    mask[
        :,
        -border:,
    ] = cv2.GC_BGD

    # Dark outline is probable foreground.
    mask[
        outline > 0
    ] = cv2.GC_PR_FGD

    # A very small dilation gives GrabCut a useful region
    # without making the entire bbox foreground.
    seed = cv2.dilate(
        outline,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (9, 9),
        ),
        iterations=1,
    )

    mask[
        seed > 0
    ] = cv2.GC_PR_FGD

    bg_model = np.zeros(
        (1, 65),
        np.float64,
    )

    fg_model = np.zeros(
        (1, 65),
        np.float64,
    )

    try:
        cv2.grabCut(
            crop,
            mask,
            None,
            bg_model,
            fg_model,
            5,
            cv2.GC_INIT_WITH_MASK,
        )

        result = np.where(
            (
                (mask == cv2.GC_FGD)
                |
                (mask == cv2.GC_PR_FGD)
            ),
            255,
            0,
        ).astype(np.uint8)

        # Keep only the largest component.
        result = largest_component(
            result
        )

        return result

    except Exception:
        return np.zeros_like(
            outline
        )


# ============================================================
# BACKGROUND PLATE
# ============================================================

def create_background_plate(
    image,
    character_mask,
):
    """
    Remove ONLY the extracted character.

    The surrounding decorations remain untouched.

    This is important for the user's scene:
    the hearts / coloured shapes should still exist in the
    restored background.
    """

    hard_mask = np.where(
        character_mask > 30,
        255,
        0,
    ).astype(np.uint8)

    h, w = hard_mask.shape

    # Remove a small halo around the character.
    dilation_size = int(
        np.clip(
            min(h, w) * 0.012,
            5,
            25,
        )
    )

    if dilation_size % 2 == 0:
        dilation_size += 1

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (
            dilation_size,
            dilation_size,
        ),
    )

    removal_mask = cv2.dilate(
        hard_mask,
        kernel,
        iterations=1,
    )

    # --------------------------------------------------------
    # Inpaint.
    # --------------------------------------------------------

    try:
        plate = cv2.inpaint(
            image,
            removal_mask,
            5,
            cv2.INPAINT_TELEA,
        )
    except Exception:
        plate = image.copy()

    # --------------------------------------------------------
    # Keep original pixels outside the removed character.
    # --------------------------------------------------------

    soft = cv2.GaussianBlur(
        removal_mask,
        (9, 9),
        0,
    ).astype(np.float32) / 255.0

    soft = soft[:, :, None]

    plate = (
        plate.astype(np.float32) * soft
        +
        image.astype(np.float32) * (1.0 - soft)
    )

    return np.clip(
        plate,
        0,
        255,
    ).astype(np.uint8)


# ============================================================
# CHARACTER CROP
# ============================================================

def crop_character(
    image,
    mask,
    padding_ratio=0.08,
):
    ys, xs = np.where(
        mask > 20
    )

    if len(xs) == 0:
        return (
            None,
            None,
            None,
            None,
        )

    x1 = int(xs.min())
    x2 = int(xs.max()) + 1

    y1 = int(ys.min())
    y2 = int(ys.max()) + 1

    width = x2 - x1
    height = y2 - y1

    px = max(
        4,
        int(width * padding_ratio),
    )

    py = max(
        4,
        int(height * padding_ratio),
    )

    x1 = max(
        0,
        x1 - px,
    )

    y1 = max(
        0,
        y1 - py,
    )

    x2 = min(
        image.shape[1],
        x2 + px,
    )

    y2 = min(
        image.shape[0],
        y2 + py,
    )

    crop = image[
        y1:y2,
        x1:x2,
    ].copy()

    alpha = mask[
        y1:y2,
        x1:x2,
    ].copy()

    center = (
        (x1 + x2) / 2.0,
        (y1 + y2) / 2.0,
    )

    return (
        crop,
        alpha,
        center,
        (x1, y1, x2, y2),
    )


# ============================================================
# ALPHA COMPOSITING
# ============================================================

def paste_layer(
    canvas,
    crop,
    alpha,
    center_x,
    center_y,
):
    if crop is None or alpha is None:
        return canvas

    h, w = canvas.shape[:2]
    ch, cw = crop.shape[:2]

    x1 = int(
        round(center_x - cw / 2)
    )

    y1 = int(
        round(center_y - ch / 2)
    )

    x2 = x1 + cw
    y2 = y1 + ch

    if (
        x2 <= 0
        or y2 <= 0
        or x1 >= w
        or y1 >= h
    ):
        return canvas

    cx1 = max(0, x1)
    cy1 = max(0, y1)

    cx2 = min(w, x2)
    cy2 = min(h, y2)

    sx1 = cx1 - x1
    sy1 = cy1 - y1

    sx2 = sx1 + (cx2 - cx1)
    sy2 = sy1 + (cy2 - cy1)

    src = crop[
        sy1:sy2,
        sx1:sx2,
    ].astype(np.float32)

    a = alpha[
        sy1:sy2,
        sx1:sx2,
    ].astype(np.float32) / 255.0

    a = a[:, :, None]

    dst = canvas[
        cy1:cy2,
        cx1:cx2,
    ].astype(np.float32)

    result = (
        src * a
        +
        dst * (1.0 - a)
    )

    canvas[
        cy1:cy2,
        cx1:cx2,
    ] = np.clip(
        result,
        0,
        255,
    ).astype(np.uint8)

    return canvas


def transform_character(
    crop,
    alpha,
    scale_x=1.0,
    scale_y=1.0,
    angle=0.0,
):
    h, w = crop.shape[:2]

    new_w = max(
        2,
        int(w * scale_x),
    )

    new_h = max(
        2,
        int(h * scale_y),
    )

    resized = cv2.resize(
        crop,
        (
            new_w,
            new_h,
        ),
        interpolation=cv2.INTER_CUBIC,
    )

    resized_alpha = cv2.resize(
        alpha,
        (
            new_w,
            new_h,
        ),
        interpolation=cv2.INTER_CUBIC,
    )

    center = (
        new_w / 2.0,
        new_h / 2.0,
    )

    matrix = cv2.getRotationMatrix2D(
        center,
        angle,
        1.0,
    )

    cos = abs(matrix[0, 0])
    sin = abs(matrix[0, 1])

    bound_w = max(
        2,
        int(
            new_h * sin
            +
            new_w * cos
        ),
    )

    bound_h = max(
        2,
        int(
            new_h * cos
            +
            new_w * sin
        ),
    )

    matrix[0, 2] += (
        bound_w / 2
        - center[0]
    )

    matrix[1, 2] += (
        bound_h / 2
        - center[1]
    )

    warped = cv2.warpAffine(
        resized,
        matrix,
        (
            bound_w,
            bound_h,
        ),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )

    warped_alpha = cv2.warpAffine(
        resized_alpha,
        matrix,
        (
            bound_w,
            bound_h,
        ),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    return (
        warped,
        warped_alpha,
    )


# ============================================================
# EASING
# ============================================================

def ease_in_out(t):
    t = float(
        np.clip(
            t,
            0.0,
            1.0,
        )
    )

    return (
        0.5
        -
        0.5
        *
        math.cos(
            math.pi * t
        )
    )


def smoothstep(t):
    t = float(
        np.clip(
            t,
            0.0,
            1.0,
        )
    )

    return (
        t * t
        *
        (
            3.0
            -
            2.0 * t
        )
    )


def lerp(a, b, t):
    return (
        a
        +
        (b - a) * t
    )


# ============================================================
# SLOW WALK MOTION
# ============================================================

def walking_pose(
    local_t,
    mode,
    step_count,
    bob_amount,
    sway_amount,
):
    phase = (
        local_t
        * step_count
        * math.pi
        * 2.0
    )

    if mode == "Walk Only":
        bob = 0.0
        sway = 0.0
        scale_y = 1.0

    elif mode == "Slow Walk + Soft Bounce":
        bob = (
            math.sin(phase)
            * bob_amount
        )

        sway = (
            math.sin(
                phase
                +
                math.pi / 2.0
            )
            * sway_amount
        )

        scale_y = (
            1.0
            +
            math.sin(phase)
            * 0.004
        )

    elif mode == "Slow Walk + Subtle Breathing":
        bob = (
            math.sin(phase)
            * bob_amount
            * 0.55
        )

        sway = (
            math.sin(
                phase * 0.5
            )
            * sway_amount
        )

        scale_y = (
            1.0
            +
            math.sin(
                phase * 0.5
            )
            * 0.009
        )

    else:
        bob = (
            math.sin(phase)
            * bob_amount
        )

        sway = (
            math.sin(
                phase
                +
                math.pi / 2.0
            )
            * sway_amount
        )

        scale_y = (
            1.0
            +
            math.sin(phase)
            * 0.003
        )

    scale_x = (
        1.0 / max(
            0.98,
            scale_y,
        )
    )

    return (
        bob,
        sway,
        scale_x,
        scale_y,
    )


# ============================================================
# FRAME RENDER
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
    merge_smoothness,
):
    h, w = original.shape[:2]

    walk_end = np.clip(
        walk_fraction,
        0.20,
        0.80,
    )

    settle_end = min(
        0.92,
        walk_end
        +
        np.clip(
            settle_fraction,
            0.03,
            0.25,
        ),
    )

    char_w = char_crop.shape[1]

    if entrance_side == "Left":
        start_x = -char_w * 0.85
    else:
        start_x = (
            w
            +
            char_w * 0.85
        )

    target_x = home_center[0]
    target_y = home_center[1]

    # --------------------------------------------------------
    # WALK
    # --------------------------------------------------------

    if t < walk_end:

        p = smoothstep(
            t
            /
            max(
                1e-6,
                walk_end,
            )
        )

        current_x = lerp(
            start_x,
            target_x,
            p,
        )

        bob, sway, sx, sy = walking_pose(
            p,
            mode,
            step_count,
            bob_amount,
            sway_amount,
        )

        current_y = (
            target_y
            +
            bob
        )

        warped_c, warped_a = transform_character(
            char_crop,
            char_alpha,
            sx,
            sy,
            sway,
        )

        frame = (
            background_plate.copy()
        )

        return paste_layer(
            frame,
            warped_c,
            warped_a,
            current_x,
            current_y,
        )

    # --------------------------------------------------------
    # SETTLE
    # --------------------------------------------------------

    if t < settle_end:

        p = smoothstep(
            (
                t
                -
                walk_end
            )
            /
            max(
                1e-6,
                settle_end
                -
                walk_end,
            )
        )

        remaining = (
            1.0 - p
        )

        bob, sway, sx, sy = walking_pose(
            remaining,
            mode,
            step_count,
            bob_amount * remaining,
            sway_amount * remaining,
        )

        current_y = (
            target_y
            +
            bob
        )

        warped_c, warped_a = transform_character(
            char_crop,
            char_alpha,
            sx,
            sy,
            sway,
        )

        frame = (
            background_plate.copy()
        )

        return paste_layer(
            frame,
            warped_c,
            warped_a,
            target_x,
            current_y,
        )

    # --------------------------------------------------------
    # MERGE INTO ORIGINAL
    # --------------------------------------------------------

    p = smoothstep(
        (
            t
            -
            settle_end
        )
        /
        max(
            1e-6,
            1.0
            -
            settle_end,
        )
    )

    # More smoothness = slower initial merge.
    exponent = np.interp(
        merge_smoothness,
        [1, 10],
        [1.0, 0.45],
    )

    merge = p ** exponent

    # Hold the character almost fully visible at the beginning
    # of the merge, then restore the real artwork.
    merge = smoothstep(
        np.clip(
            merge * 1.15,
            0.0,
            1.0,
        )
    )

    moving = (
        background_plate.copy()
    )

    character_factor = (
        1.0
        -
        merge
    )

    warped_c, warped_a = transform_character(
        char_crop,
        char_alpha,
        1.0,
        1.0,
        0.0,
    )

    warped_a = np.clip(
        warped_a.astype(
            np.float32
        )
        *
        character_factor,
        0,
        255,
    ).astype(np.uint8)

    moving = paste_layer(
        moving,
        warped_c,
        warped_a,
        target_x,
        target_y,
    )

    frame = (
        moving.astype(
            np.float32
        )
        *
        (1.0 - merge)
        +
        original.astype(
            np.float32
        )
        *
        merge
    )

    return np.clip(
        frame,
        0,
        255,
    ).astype(np.uint8)


# ============================================================
# EXPORT
# ============================================================

def build_gif(
    frames,
    fps,
):
    buffer = io.BytesIO()

    prepared = [
        Image.fromarray(
            cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB,
            )
        )
        for frame in frames
    ]

    duration = max(
        20,
        int(
            1000
            /
            max(
                1,
                fps,
            )
        ),
    )

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


def build_apng(
    frames,
    fps,
):
    buffer = io.BytesIO()

    prepared = [
        Image.fromarray(
            cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB,
            )
        ).convert("RGB")
        for frame in frames
    ]

    duration = max(
        20,
        int(
            1000
            /
            max(
                1,
                fps,
            )
        ),
    )

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
    options=[
        8,
        10,
        12,
        15,
        20,
        24,
    ],
    value=12,
)

duration = st.sidebar.slider(
    "Total duration (seconds)",
    5.0,
    16.0,
    9.0,
    0.5,
)

st.sidebar.markdown("---")

st.sidebar.header("🚶 Slow Walk")

walk_percent = st.sidebar.slider(
    "Walking time (%)",
    25,
    65,
    45,
)

settle_percent = st.sidebar.slider(
    "Settle time (%)",
    5,
    25,
    12,
)

entrance_side = st.sidebar.selectbox(
    "Entrance side",
    [
        "Left",
        "Right",
    ],
)

step_count = st.sidebar.slider(
    "Walking steps",
    2,
    12,
    4,
)

bob_amount = st.sidebar.slider(
    "Vertical motion",
    0.0,
    10.0,
    2.5,
    0.5,
)

sway_amount = st.sidebar.slider(
    "Body sway",
    0.0,
    5.0,
    1.2,
    0.25,
)

motion_mode = st.sidebar.selectbox(
    "Motion style",
    MOTION_MODES,
)

st.sidebar.markdown("---")

st.sidebar.header("✂️ Clean Extraction")

outline_sensitivity = st.sidebar.slider(
    "Outline sensitivity",
    0,
    100,
    55,
    help=(
        "Higher values detect lighter/broken hand-drawn outlines. "
        "If paper texture becomes selected, lower this."
    ),
)

detail_strength = st.sidebar.slider(
    "Interior colour/detail",
    0,
    100,
    45,
    help=(
        "Controls how much coloured/pencil detail is retained "
        "inside the detected character silhouette."
    ),
)

padding = st.sidebar.slider(
    "Character edge padding (%)",
    2,
    20,
    8,
)

st.sidebar.markdown(
    """
**Recommended for your llama drawing**

- Outline sensitivity: **45–60**
- Interior colour/detail: **35–50**
- Bounding box: tightly around the llama
- Walking steps: **4**
- Vertical motion: **2–3**
- Body sway: **1–1.5**
- Duration: **9–10 seconds**
"""
)

st.sidebar.markdown("---")

st.sidebar.header("🌅 Scene Merge")

merge_smoothness = st.sidebar.slider(
    "Merge smoothness",
    1,
    10,
    7,
)

st.sidebar.markdown("---")

st.sidebar.header("✨ Optional Colour Polish")

enable_enhancement = st.sidebar.checkbox(
    "Enable mild warmth",
    value=False,
)

temp_shift = st.sidebar.slider(
    "Warmth",
    0,
    20,
    5,
)


# ============================================================
# UPLOAD
# ============================================================

uploaded_files = st.file_uploader(
    "Upload one or more drawings",
    type=[
        "jpg",
        "jpeg",
        "png",
        "webp",
    ],
    accept_multiple_files=True,
)

if not uploaded_files:
    st.info(
        "Upload a drawing. Put the X/Y bounding box around the "
        "character, not around the whole scene."
    )
    st.stop()


# ============================================================
# PROCESS
# ============================================================

zip_export_files = {}

for idx, uploaded in enumerate(
    uploaded_files
):

    st.markdown("---")

    st.subheader(
        f"🖼️ Drawing {idx + 1}: {uploaded.name}"
    )

    uploaded.seek(0)

    file_bytes = np.asarray(
        bytearray(
            uploaded.read()
        ),
        dtype=np.uint8,
    )

    raw = cv2.imdecode(
        file_bytes,
        cv2.IMREAD_COLOR,
    )

    if raw is None:
        st.error(
            f"Could not read {uploaded.name}."
        )
        continue

    raw = auto_rotate_vertical(
        raw
    )

    if enable_enhancement:
        image = enhance_color_temperature_and_warmth(
            raw,
            temp_shift=temp_shift,
            saturation_boost=1.03,
        )
    else:
        image = raw.copy()

    image = resize_image(
        image
    )

    # --------------------------------------------------------
    # BOUNDING BOX
    # --------------------------------------------------------

    c1, c2 = st.columns(2)

    with c1:
        x_range = st.slider(
            f"Character X range #{idx + 1}",
            0,
            100,
            (15, 85),
            key=f"x_{idx}_{uploaded.name}",
        )

    with c2:
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
    # EXTRACT
    # --------------------------------------------------------

    mask_preview, bbox = extract_character_mask(
        image,
        bbox_pct,
        outline_sensitivity=outline_sensitivity,
        detail_strength=detail_strength,
    )

    char_preview, alpha_preview, home_preview, char_bbox = crop_character(
        image,
        mask_preview,
        padding_ratio=padding / 100.0,
    )

    # --------------------------------------------------------
    # PREVIEWS
    # --------------------------------------------------------

    p1, p2, p3 = st.columns(3)

    with p1:
        st.image(
            cv2.cvtColor(
                image,
                cv2.COLOR_BGR2RGB,
            ),
            caption="Original artwork",
            width="stretch",
        )

    with p2:

        overlay = image.copy()

        red = np.zeros_like(
            image
        )

        # OpenCV BGR: channel 2 is red.
        red[:, :, 2] = 255

        m = (
            mask_preview.astype(
                np.float32
            )
            /
            255.0
        )

        m = m[:, :, None]

        overlay = (
            overlay.astype(
                np.float32
            )
            *
            (1.0 - 0.52 * m)
            +
            red.astype(
                np.float32
            )
            *
            (0.52 * m)
        )

        overlay = np.clip(
            overlay,
            0,
            255,
        ).astype(np.uint8)

        st.image(
            cv2.cvtColor(
                overlay,
                cv2.COLOR_BGR2RGB,
            ),
            caption=(
                "Extraction preview — red = character"
            ),
            width="stretch",
        )

    with p3:

        if char_preview is not None:

            white = np.ones_like(
                char_preview
            ) * 255

            a = (
                alpha_preview.astype(
                    np.float32
                )
                /
                255.0
            )

            a = a[:, :, None]

            isolated = (
                char_preview.astype(
                    np.float32
                )
                *
                a
                +
                white.astype(
                    np.float32
                )
                *
                (1.0 - a)
            )

            isolated = np.clip(
                isolated,
                0,
                255,
            ).astype(np.uint8)

            st.image(
                cv2.cvtColor(
                    isolated,
                    cv2.COLOR_BGR2RGB,
                ),
                caption="Isolated character",
                width="stretch",
            )

        else:
            st.error(
                "No character was detected."
            )

    # --------------------------------------------------------
    # MASK QUALITY INDICATOR
    # --------------------------------------------------------

    selected_ratio = (
        np.count_nonzero(
            mask_preview > 30
        )
        /
        max(
            1,
            image.shape[0]
            *
            image.shape[1],
        )
    )

    if selected_ratio > 0.35:
        st.warning(
            "⚠️ The selected area is very large. "
            "Tighten the X/Y bounding box or lower Outline sensitivity."
        )

    elif selected_ratio < 0.003:
        st.warning(
            "⚠️ Very little was selected. "
            "Increase Outline sensitivity or slightly widen the bounding box."
        )

    else:
        st.success(
            f"✓ Character mask size looks reasonable "
            f"({selected_ratio * 100:.1f}% of image)."
        )

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
            "The character could not be isolated."
        )
        continue

    # --------------------------------------------------------
    # BACKGROUND PLATE
    # --------------------------------------------------------

    with st.spinner(
        "Creating clean background plate..."
    ):
        background_plate = create_background_plate(
            image,
            mask_preview,
        )

    st.image(
        cv2.cvtColor(
            background_plate,
            cv2.COLOR_BGR2RGB,
        ),
        caption=(
            "Background plate — only the character removed"
        ),
        width="stretch",
    )

    # --------------------------------------------------------
    # CHARACTER
    # --------------------------------------------------------

    char_crop, char_alpha, home_center, char_bbox = crop_character(
        image,
        mask_preview,
        padding_ratio=padding / 100.0,
    )

    if char_crop is None:
        st.error(
            "Character crop failed."
        )
        continue

    frame_count = max(
        12,
        int(
            round(
                fps
                *
                duration
            )
        ),
    )

    walk_fraction = (
        walk_percent
        /
        100.0
    )

    settle_fraction = (
        settle_percent
        /
        100.0
    )

    progress = st.progress(
        0,
        text="Rendering slow walk...",
    )

    frames = []

    for i in range(
        frame_count
    ):

        t = (
            i
            /
            max(
                1,
                frame_count - 1,
            )
        )

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
            merge_smoothness=merge_smoothness,
        )

        frames.append(frame)

        progress.progress(
            (i + 1)
            /
            frame_count,
            text=(
                f"Rendering frame "
                f"{i + 1}/{frame_count}"
            ),
        )

    progress.empty()

    # --------------------------------------------------------
    # EXPORT
    # --------------------------------------------------------

    gif_data = build_gif(
        frames,
        fps,
    )

    apng_data = build_apng(
        frames,
        fps,
    )

    base_name = os.path.splitext(
        uploaded.name
    )[0]

    gif_name = (
        f"natural_walk_{base_name}.gif"
    )

    apng_name = (
        f"natural_walk_{base_name}.png"
    )

    zip_export_files[
        gif_name
    ] = gif_data

    zip_export_files[
        apng_name
    ] = apng_data

    st.markdown(
        "### 🎬 Final animation"
    )

    st.image(
        gif_data,
        caption=(
            "Slow walk → settle → merge into original artwork"
        ),
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
# ZIP
# ============================================================

if zip_export_files:

    st.markdown("---")

    st.subheader(
        "📦 Bulk Download"
    )

    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(
        zip_buffer,
        "w",
        zipfile.ZIP_DEFLATED,
    ) as z:

        for name, data in zip_export_files.items():
            z.writestr(
                name,
                data,
            )

    st.download_button(
        "📦 Download All Animations",
        data=zip_buffer.getvalue(),
        file_name="natural_walk_animations.zip",
        mime="application/zip",
        type="primary",
        width="stretch",
    )


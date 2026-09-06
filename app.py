import io
import math
import os
import tempfile

import cv2
import numpy as np
import streamlit as st
from PIL import Image


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Hand-Drawn Walk Away Animator",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🎨 Hand-Drawn Character Walk-Away Animator")

st.markdown(
    """
Turn a single hand-drawn character into a simple frame-by-frame animation.

**The character is separated from the background, then walks away from the
canvas until it completely disappears.**

No AI model is required.
"""
)


# ============================================================
# CONSTANTS
# ============================================================

MAX_IMAGE_SIZE = 1100

DIRECTIONS = [
    "Right",
    "Left",
    "Down",
    "Up",
    "Diagonal Down-Right",
    "Diagonal Down-Left",
    "Diagonal Up-Right",
    "Diagonal Up-Left",
]

BACKGROUND_MODES = [
    "White / Light Paper",
    "Dark Background",
    "Automatic",
]


# ============================================================
# IMAGE HELPERS
# ============================================================

def resize_image(image, max_size=MAX_IMAGE_SIZE):
    h, w = image.shape[:2]

    if max(h, w) <= max_size:
        return image.copy()

    scale = max_size / float(max(h, w))

    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))

    return cv2.resize(
        image,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA,
    )


def bgr_to_rgb(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


# ============================================================
# BACKGROUND ESTIMATION
# ============================================================

def get_border_pixels(image, border_size=12):
    """
    Collect pixels from the outside border of the image.
    These are assumed to mostly represent the paper/background.
    """

    h, w = image.shape[:2]

    border_size = max(
        2,
        min(border_size, h // 4, w // 4)
    )

    top = image[:border_size, :, :]
    bottom = image[h - border_size:h, :, :]
    left = image[:, :border_size, :]
    right = image[:, w - border_size:w, :]

    pixels = np.concatenate(
        [
            top.reshape(-1, 3),
            bottom.reshape(-1, 3),
            left.reshape(-1, 3),
            right.reshape(-1, 3),
        ],
        axis=0,
    )

    return pixels


def estimate_background_lab(image):
    """
    Estimate background color using the image border.
    """

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    border = get_border_pixels(image)

    border_lab = cv2.cvtColor(
        border.reshape(-1, 1, 3),
        cv2.COLOR_BGR2LAB
    ).reshape(-1, 3)

    median_lab = np.median(
        border_lab,
        axis=0
    ).astype(np.float32)

    return median_lab


# ============================================================
# FOREGROUND EXTRACTION
# ============================================================

def normalize_map(values):
    values = values.astype(np.float32)

    mn = float(np.min(values))
    mx = float(np.max(values))

    if mx - mn < 1e-6:
        return np.zeros_like(values)

    result = (values - mn) / (mx - mn)

    return np.clip(result, 0.0, 1.0)


def build_foreground_mask(
    image,
    sensitivity=50,
    background_mode="White / Light Paper",
):
    """
    Extract the complete hand-drawn character.

    This combines:

    - difference from estimated paper color
    - luminance difference
    - black-hat morphology
    - local contrast
    - edges
    """

    h, w = image.shape[:2]

    # --------------------------------------------------------
    # LAB COLOR DIFFERENCE
    # --------------------------------------------------------

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    bg_lab = estimate_background_lab(image)

    diff = lab.astype(np.float32) - bg_lab.reshape(1, 1, 3)

    color_distance = np.sqrt(
        np.sum(diff * diff, axis=2)
    )

    color_distance = normalize_map(color_distance)

    # --------------------------------------------------------
    # GRAYSCALE / DARKNESS
    # --------------------------------------------------------

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    border = get_border_pixels(image)

    border_gray = cv2.cvtColor(
        border.reshape(-1, 1, 3),
        cv2.COLOR_BGR2GRAY
    ).reshape(-1)

    bg_gray = float(np.median(border_gray))

    if background_mode == "Dark Background":
        darkness = gray.astype(np.float32) - bg_gray
    else:
        darkness = bg_gray - gray.astype(np.float32)

    darkness = np.clip(
        darkness,
        0,
        None
    )

    darkness = normalize_map(darkness)

    # --------------------------------------------------------
    # LOCAL CONTRAST
    # --------------------------------------------------------

    local_background = cv2.GaussianBlur(
        gray,
        (0, 0),
        sigmaX=max(5, min(h, w) / 80)
    )

    if background_mode == "Dark Background":
        local_difference = (
            gray.astype(np.float32)
            - local_background.astype(np.float32)
        )
    else:
        local_difference = (
            local_background.astype(np.float32)
            - gray.astype(np.float32)
        )

    local_difference = np.clip(
        local_difference,
        0,
        None
    )

    local_difference = normalize_map(
        local_difference
    )

    # --------------------------------------------------------
    # BLACK HAT
    # --------------------------------------------------------

    kernel_size = max(
        9,
        int(min(h, w) * 0.025)
    )

    if kernel_size % 2 == 0:
        kernel_size += 1

    blackhat_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_size, kernel_size)
    )

    blackhat = cv2.morphologyEx(
        gray,
        cv2.MORPH_BLACKHAT,
        blackhat_kernel
    )

    blackhat = normalize_map(blackhat)

    # --------------------------------------------------------
    # EDGE STRUCTURE
    # --------------------------------------------------------

    blurred = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    edges = cv2.Canny(
        blurred,
        30,
        100
    )

    edges = cv2.dilate(
        edges,
        np.ones((3, 3), np.uint8),
        iterations=1
    )

    edges = edges.astype(np.float32) / 255.0

    # --------------------------------------------------------
    # COMBINE SIGNALS
    # --------------------------------------------------------

    score = (
        color_distance * 0.40
        + darkness * 0.28
        + local_difference * 0.12
        + blackhat * 0.15
        + edges * 0.05
    )

    score = cv2.GaussianBlur(
        score,
        (5, 5),
        0
    )

    # --------------------------------------------------------
    # THRESHOLD
    # --------------------------------------------------------

    # Lower sensitivity = easier extraction
    # Higher sensitivity = more selective
    percentile = np.clip(
        88 - sensitivity * 0.35,
        65,
        90
    )

    threshold = np.percentile(
        score,
        percentile
    )

    mask = (
        score >= threshold
    ).astype(np.uint8) * 255

    # --------------------------------------------------------
    # RECOVER STRONG DARK/COLORED PIXELS
    # --------------------------------------------------------

    strong_signal = (
        color_distance > 0.22
    ) | (
        darkness > 0.20
    ) | (
        blackhat > 0.30
    )

    mask[
        strong_signal
        & (score > threshold * 0.55)
    ] = 255

    # --------------------------------------------------------
    # MORPHOLOGY
    # --------------------------------------------------------

    close_kernel = np.ones(
        (5, 5),
        np.uint8
    )

    open_kernel = np.ones(
        (3, 3),
        np.uint8
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        close_kernel,
        iterations=2
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        open_kernel,
        iterations=1
    )

    # --------------------------------------------------------
    # CONNECTED COMPONENT FILTERING
    # --------------------------------------------------------

    num_labels, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            mask,
            connectivity=8
        )
    )

    if num_labels <= 1:
        return mask

    total_area = h * w

    components = []

    for i in range(1, num_labels):

        area = int(
            stats[i, cv2.CC_STAT_AREA]
        )

        if area < max(
            20,
            int(total_area * 0.00003)
        ):
            continue

        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        cw = int(stats[i, cv2.CC_STAT_WIDTH])
        ch = int(stats[i, cv2.CC_STAT_HEIGHT])

        cx, cy = centroids[i]

        components.append(
            {
                "label": i,
                "area": area,
                "x": x,
                "y": y,
                "w": cw,
                "h": ch,
                "cx": cx,
                "cy": cy,
            }
        )

    if not components:
        return mask

    components.sort(
        key=lambda x: x["area"],
        reverse=True
    )

    # --------------------------------------------------------
    # CHARACTER COMPONENT SELECTION
    # --------------------------------------------------------

    largest_area = components[0]["area"]

    selected = []

    center_x = w / 2
    center_y = h / 2

    for comp in components:

        area_ratio = (
            comp["area"]
            / max(1, largest_area)
        )

        distance = math.sqrt(
            (
                comp["cx"] - center_x
            ) ** 2
            +
            (
                comp["cy"] - center_y
            ) ** 2
        )

        normalized_distance = (
            distance
            /
            max(1, math.sqrt(w * w + h * h))
        )

        # Keep large components.
        if area_ratio >= 0.08:
            selected.append(comp["label"])
            continue

        # Keep reasonably large components close
        # to the main character.
        if (
            area_ratio >= 0.015
            and normalized_distance < 0.35
        ):
            selected.append(comp["label"])

    if not selected:
        selected = [
            components[0]["label"]
        ]

    clean_mask = np.zeros_like(mask)

    for label in selected:
        clean_mask[
            labels == label
        ] = 255

    # --------------------------------------------------------
    # CONTOUR FILL
    # --------------------------------------------------------

    contours, _ = cv2.findContours(
        clean_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    filled = np.zeros_like(
        clean_mask
    )

    for contour in contours:

        area = cv2.contourArea(
            contour
        )

        if area < 20:
            continue

        cv2.drawContours(
            filled,
            [contour],
            -1,
            255,
            thickness=cv2.FILLED
        )

    # Combine original fine details with filled body.
    combined = cv2.bitwise_or(
        clean_mask,
        filled
    )

    # --------------------------------------------------------
    # FINAL SMOOTHING
    # --------------------------------------------------------

    combined = cv2.morphologyEx(
        combined,
        cv2.MORPH_CLOSE,
        np.ones((5, 5), np.uint8),
        iterations=1
    )

    return combined


# ============================================================
# MASK REFINEMENT
# ============================================================

def refine_mask(mask, feather=2):
    mask = mask.astype(np.uint8)

    # Remove tiny holes.
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        np.ones((5, 5), np.uint8),
        iterations=1
    )

    # Small dilation prevents cutting off outlines.
    mask = cv2.dilate(
        mask,
        np.ones((3, 3), np.uint8),
        iterations=1
    )

    # Feather edge.
    if feather > 0:
        kernel = feather * 2 + 1

        mask = cv2.GaussianBlur(
            mask,
            (kernel, kernel),
            0
        )

    return mask


# ============================================================
# CHARACTER EXTRACTION
# ============================================================

def extract_character(
    image,
    mask,
    margin=20
):
    """
    Return cropped BGR image + alpha mask.
    """

    ys, xs = np.where(
        mask > 20
    )

    if len(xs) == 0:
        return None, None, None

    x1 = max(
        0,
        int(np.min(xs)) - margin
    )

    y1 = max(
        0,
        int(np.min(ys)) - margin
    )

    x2 = min(
        image.shape[1],
        int(np.max(xs)) + margin + 1
    )

    y2 = min(
        image.shape[0],
        int(np.max(ys)) + margin + 1
    )

    crop = image[
        y1:y2,
        x1:x2
    ].copy()

    alpha = mask[
        y1:y2,
        x1:x2
    ].copy()

    return crop, alpha, (
        x1,
        y1,
        x2,
        y2
    )


# ============================================================
# BACKGROUND RECONSTRUCTION
# ============================================================

def reconstruct_background(
    image,
    mask,
    strength=7
):
    """
    Remove the character from its original location
    using OpenCV inpainting.
    """

    inpaint_mask = mask.copy()

    inpaint_mask = cv2.dilate(
        inpaint_mask,
        np.ones((7, 7), np.uint8),
        iterations=2
    )

    # Don't attempt absurdly large inpainting areas.
    coverage = (
        np.count_nonzero(inpaint_mask)
        /
        float(inpaint_mask.size)
    )

    if coverage > 0.65:
        return image.copy()

    background = cv2.inpaint(
        image,
        inpaint_mask,
        strength,
        cv2.INPAINT_TELEA
    )

    return background


# ============================================================
# TRANSPARENT CHARACTER PREVIEW
# ============================================================

def checkerboard(
    width,
    height,
    square=20
):
    result = np.zeros(
        (height, width, 3),
        dtype=np.uint8
    )

    for y in range(0, height, square):
        for x in range(0, width, square):

            if (
                (x // square + y // square)
                % 2
                == 0
            ):
                value = 225
            else:
                value = 245

            result[
                y:min(y + square, height),
                x:min(x + square, width)
            ] = value

    return result


def composite_character_preview(
    character,
    alpha
):
    h, w = character.shape[:2]

    bg = checkerboard(
        w,
        h
    )

    a = (
        alpha.astype(np.float32)
        / 255.0
    )

    a = a[:, :, None]

    result = (
        character.astype(np.float32)
        * a
        +
        bg.astype(np.float32)
        * (1 - a)
    )

    return np.clip(
        result,
        0,
        255
    ).astype(np.uint8)


# ============================================================
# EASING
# ============================================================

def smoothstep(t):
    t = np.clip(
        t,
        0.0,
        1.0
    )

    return (
        t * t * (3 - 2 * t)
    )


def ease_in_out(t):
    t = np.clip(
        t,
        0.0,
        1.0
    )

    return (
        0.5
        -
        0.5
        *
        math.cos(math.pi * t)
    )


# ============================================================
# WALK TRAJECTORY
# ============================================================

def get_start_center(
    bbox,
    canvas_w,
    canvas_h
):
    x1, y1, x2, y2 = bbox

    return (
        (x1 + x2) / 2,
        (y1 + y2) / 2
    )


def get_exit_center(
    start_x,
    start_y,
    char_w,
    char_h,
    canvas_w,
    canvas_h,
    direction
):
    margin_x = char_w * 1.4
    margin_y = char_h * 1.4

    if direction == "Right":
        return (
            canvas_w + margin_x,
            start_y
        )

    if direction == "Left":
        return (
            -margin_x,
            start_y
        )

    if direction == "Down":
        return (
            start_x,
            canvas_h + margin_y
        )

    if direction == "Up":
        return (
            start_x,
            -margin_y
        )

    if direction == "Diagonal Down-Right":
        return (
            canvas_w + margin_x,
            canvas_h + margin_y
        )

    if direction == "Diagonal Down-Left":
        return (
            -margin_x,
            canvas_h + margin_y
        )

    if direction == "Diagonal Up-Right":
        return (
            canvas_w + margin_x,
            -margin_y
        )

    if direction == "Diagonal Up-Left":
        return (
            -margin_x,
            -margin_y
        )

    return (
        canvas_w + margin_x,
        start_y
    )


def walking_position(
    t,
    start,
    exit_pos,
    bob_amount,
    sway_amount,
    cycles
):
    """
    Calculate character center for a frame.
    """

    motion_t = ease_in_out(t)

    sx, sy = start
    ex, ey = exit_pos

    x = (
        sx
        +
        (ex - sx)
        * motion_t
    )

    y = (
        sy
        +
        (ey - sy)
        * motion_t
    )

    # Walking bob.
    # It gradually becomes less noticeable near the end.
    bob_fade = 1.0 - smoothstep(
        max(0.0, (t - 0.75) / 0.25)
    )

    bob = (
        math.sin(
            2
            * math.pi
            * cycles
            * t
        )
        * bob_amount
        * bob_fade
    )

    y += bob

    # Small body sway.
    sway = (
        math.sin(
            2
            * math.pi
            * cycles
            * t
            +
            math.pi / 2
        )
        *
        sway_amount
        *
        bob_fade
    )

    return x, y, sway


# ============================================================
# CHARACTER TRANSFORMATION
# ============================================================

def transform_character(
    character,
    alpha,
    scale,
    angle
):
    h, w = character.shape[:2]

    new_w = max(
        2,
        int(w * scale)
    )

    new_h = max(
        2,
        int(h * scale)
    )

    resized = cv2.resize(
        character,
        (new_w, new_h),
        interpolation=cv2.INTER_LINEAR
    )

    resized_alpha = cv2.resize(
        alpha,
        (new_w, new_h),
        interpolation=cv2.INTER_LINEAR
    )

    center = (
        new_w / 2,
        new_h / 2
    )

    matrix = cv2.getRotationMatrix2D(
        center,
        angle,
        1.0
    )

    cos = abs(matrix[0, 0])
    sin = abs(matrix[0, 1])

    bound_w = int(
        new_h * sin
        +
        new_w * cos
    )

    bound_h = int(
        new_h * cos
        +
        new_w * sin
    )

    matrix[0, 2] += (
        bound_w / 2
        -
        center[0]
    )

    matrix[1, 2] += (
        bound_h / 2
        -
        center[1]
    )

    rotated = cv2.warpAffine(
        resized,
        matrix,
        (bound_w, bound_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255)
    )

    rotated_alpha = cv2.warpAffine(
        resized_alpha,
        matrix,
        (bound_w, bound_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )

    return rotated, rotated_alpha


# ============================================================
# COMPOSITING
# ============================================================

def paste_character(
    background,
    character,
    alpha,
    center_x,
    center_y,
    fade=1.0
):
    canvas = background.copy()

    h, w = character.shape[:2]

    x1 = int(
        round(center_x - w / 2)
    )

    y1 = int(
        round(center_y - h / 2)
    )

    x2 = x1 + w
    y2 = y1 + h

    canvas_h, canvas_w = canvas.shape[:2]

    # Completely outside.
    if (
        x2 <= 0
        or
        y2 <= 0
        or
        x1 >= canvas_w
        or
        y1 >= canvas_h
    ):
        return canvas

    # Clip to canvas.
    cx1 = max(0, x1)
    cy1 = max(0, y1)
    cx2 = min(canvas_w, x2)
    cy2 = min(canvas_h, y2)

    src_x1 = cx1 - x1
    src_y1 = cy1 - y1
    src_x2 = src_x1 + (cx2 - cx1)
    src_y2 = src_y1 + (cy2 - cy1)

    char_crop = character[
        src_y1:src_y2,
        src_x1:src_x2
    ]

    alpha_crop = alpha[
        src_y1:src_y2,
        src_x1:src_x2
    ]

    a = (
        alpha_crop.astype(np.float32)
        / 255.0
    )

    a *= float(
        np.clip(
            fade,
            0,
            1
        )
    )

    a = a[:, :, None]

    background_crop = canvas[
        cy1:cy2,
        cx1:cx2
    ].astype(np.float32)

    foreground = (
        char_crop.astype(np.float32)
        * a
    )

    result = (
        foreground
        +
        background_crop
        *
        (1 - a)
    )

    canvas[
        cy1:cy2,
        cx1:cx2
    ] = np.clip(
        result,
        0,
        255
    ).astype(np.uint8)

    return canvas


# ============================================================
# RENDER ONE FRAME
# ============================================================

def render_walk_frame(
    background,
    character,
    alpha,
    t,
    start,
    exit_pos,
    end_scale,
    bob_amount,
    sway_amount,
    cycles,
    fade_at_end,
):
    canvas_h, canvas_w = background.shape[:2]

    x, y, walking_sway = walking_position(
        t,
        start,
        exit_pos,
        bob_amount,
        sway_amount,
        cycles
    )

    # Scale changes from 100% to user-selected end scale.
    scale = (
        1.0
        +
        (
            end_scale
            - 1.0
        )
        * ease_in_out(t)
    )

    # Slight walking rotation.
    angle = walking_sway

    transformed_character, transformed_alpha = (
        transform_character(
            character,
            alpha,
            scale,
            angle
        )
    )

    # Optional disappearance fade.
    if fade_at_end and t > 0.82:
        fade = 1.0 - (
            (t - 0.82)
            / 0.18
        )
    else:
        fade = 1.0

    return paste_character(
        background,
        transformed_character,
        transformed_alpha,
        x,
        y,
        fade
    )


# ============================================================
# CONTACT SHEET
# ============================================================

def make_contact_sheet(
    frames,
    columns=4,
    max_width=1000
):
    if not frames:
        return None

    thumbs = []

    for frame in frames:
        h, w = frame.shape[:2]

        thumb_w = 260
        thumb_h = max(
            1,
            int(
                h
                *
                thumb_w
                /
                max(1, w)
            )
        )

        thumb = cv2.resize(
            frame,
            (thumb_w, thumb_h),
            interpolation=cv2.INTER_AREA
        )

        thumbs.append(thumb)

    rows = math.ceil(
        len(thumbs) / columns
    )

    cell_h = max(
        x.shape[0]
        for x in thumbs
    )

    cell_w = max(
        x.shape[1]
        for x in thumbs
    )

    sheet = np.ones(
        (
            rows * cell_h,
            columns * cell_w,
            3
        ),
        dtype=np.uint8
    ) * 255

    for i, thumb in enumerate(thumbs):

        row = i // columns
        col = i % columns

        y = row * cell_h
        x = col * cell_w

        sheet[
            y:y + thumb.shape[0],
            x:x + thumb.shape[1]
        ] = thumb

    if sheet.shape[1] > max_width:

        scale = (
            max_width
            /
            sheet.shape[1]
        )

        sheet = cv2.resize(
            sheet,
            (
                int(sheet.shape[1] * scale),
                int(sheet.shape[0] * scale)
            ),
            interpolation=cv2.INTER_AREA
        )

    return sheet


# ============================================================
# GIF EXPORT
# ============================================================

def build_gif(
    frames,
    fps
):
    if not frames:
        return None

    buffer = io.BytesIO()

    prepared = []

    duration = max(
        20,
        int(1000 / fps)
    )

    for frame in frames:

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        pil = Image.fromarray(
            rgb
        ).convert("P", palette=Image.ADAPTIVE)

        prepared.append(pil)

    prepared[0].save(
        buffer,
        format="GIF",
        save_all=True,
        append_images=prepared[1:],
        duration=duration,
        loop=0,
        optimize=False,
    )

    return buffer.getvalue()


# ============================================================
# MP4 EXPORT
# ============================================================

def build_mp4(
    frames,
    fps
):
    if not frames:
        return None

    h, w = frames[0].shape[:2]

    with tempfile.NamedTemporaryFile(
        suffix=".mp4",
        delete=False
    ) as tmp:

        temp_path = tmp.name

    try:

        fourcc = cv2.VideoWriter_fourcc(
            *"mp4v"
        )

        writer = cv2.VideoWriter(
            temp_path,
            fourcc,
            fps,
            (w, h)
        )

        if not writer.isOpened():
            return None

        for frame in frames:
            writer.write(frame)

        writer.release()

        with open(
            temp_path,
            "rb"
        ) as f:
            return f.read()

    except Exception:
        return None

    finally:

        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("🎬 Animation")

direction = st.sidebar.selectbox(
    "Walk Direction",
    DIRECTIONS,
    index=0
)

fps = st.sidebar.select_slider(
    "FPS",
    options=[
        8,
        10,
        12,
        15,
        20,
        24,
        30,
    ],
    value=12
)

duration = st.sidebar.slider(
    "Animation Duration",
    min_value=1.0,
    max_value=8.0,
    value=3.0,
    step=0.5
)

st.sidebar.markdown("---")

st.sidebar.header("🚶 Walking Motion")

bob_amount = st.sidebar.slider(
    "Walking Bob",
    min_value=0,
    max_value=30,
    value=5
)

sway_amount = st.sidebar.slider(
    "Walking Sway",
    min_value=0,
    max_value=15,
    value=3
)

cycles = st.sidebar.slider(
    "Walking Steps",
    min_value=1,
    max_value=12,
    value=5
)

st.sidebar.markdown("---")

st.sidebar.header("📏 Walking Away")

end_scale_percent = st.sidebar.slider(
    "Final Character Size",
    min_value=20,
    max_value=100,
    value=100,
    step=5
)

st.sidebar.caption(
    "100% = same size while walking out. "
    "Smaller values create a stronger 'walking away into distance' effect."
)

fade_at_end = st.sidebar.checkbox(
    "Fade slightly at the very end",
    value=False
)

st.sidebar.markdown("---")

st.sidebar.header("🎨 Character Extraction")

background_mode = st.sidebar.selectbox(
    "Background Type",
    BACKGROUND_MODES,
    index=0
)

sensitivity = st.sidebar.slider(
    "Foreground Sensitivity",
    min_value=20,
    max_value=80,
    value=50
)

st.sidebar.caption(
    "Increase sensitivity if too much background is being selected. "
    "Decrease it if parts of the character are missing."
)


# ============================================================
# FILE UPLOAD
# ============================================================

uploaded = st.file_uploader(
    "Upload your hand-drawn character",
    type=[
        "jpg",
        "jpeg",
        "png",
        "webp"
    ]
)


# ============================================================
# MAIN PROCESSING
# ============================================================

if uploaded is not None:

    try:

        # ----------------------------------------------------
        # READ IMAGE
        # ----------------------------------------------------

        file_bytes = np.asarray(
            bytearray(
                uploaded.read()
            ),
            dtype=np.uint8
        )

        image = cv2.imdecode(
            file_bytes,
            cv2.IMREAD_COLOR
        )

        if image is None:
            st.error(
                "Could not read the uploaded image."
            )
            st.stop()

        image = resize_image(
            image,
            MAX_IMAGE_SIZE
        )

        h, w = image.shape[:2]

        # ----------------------------------------------------
        # EXTRACT CHARACTER
        # ----------------------------------------------------

        with st.spinner(
            "🔍 Separating character from background..."
        ):

            mask = build_foreground_mask(
                image,
                sensitivity=sensitivity,
                background_mode=background_mode
            )

            mask = refine_mask(
                mask,
                feather=2
            )

            character, alpha, bbox = (
                extract_character(
                    image,
                    mask,
                    margin=20
                )
            )

        if character is None:

            st.error(
                "❌ I could not detect a character in this image."
            )

            st.info(
                "Try lowering Foreground Sensitivity or "
                "using a clearer image with a light background."
            )

            st.image(
                bgr_to_rgb(image),
                use_container_width=True
            )

            st.stop()

        # ----------------------------------------------------
        # CHARACTER SIZE CHECK
        # ----------------------------------------------------

        char_h, char_w = character.shape[:2]

        character_area = (
            char_w
            *
            char_h
        )

        image_area = (
            w
            *
            h
        )

        if character_area > image_area * 0.95:

            st.warning(
                "⚠️ The detected character occupies almost "
                "the entire image. Background separation "
                "may need adjustment."
            )

        # ----------------------------------------------------
        # BACKGROUND RECONSTRUCTION
        # ----------------------------------------------------

        with st.spinner(
            "🧹 Reconstructing background..."
        ):

            background = reconstruct_background(
                image,
                mask,
                strength=7
            )

        # ----------------------------------------------------
        # PREVIEWS
        # ----------------------------------------------------

        st.subheader(
            "🔍 Extraction Preview"
        )

        c1, c2, c3 = st.columns(3)

        with c1:

            st.markdown(
                "**Original**"
            )

            st.image(
                bgr_to_rgb(image),
                use_container_width=True
            )

        with c2:

            st.markdown(
                "**Detected Character**"
            )

            preview = composite_character_preview(
                character,
                alpha
            )

            st.image(
                bgr_to_rgb(preview),
                use_container_width=True
            )

        with c3:

            st.markdown(
                "**Clean Background**"
            )

            st.image(
                bgr_to_rgb(background),
                use_container_width=True
            )

        # ----------------------------------------------------
        # EXTRACTION QUALITY INFORMATION
        # ----------------------------------------------------

        st.markdown("---")

        info1, info2, info3, info4 = st.columns(4)

        with info1:
            st.metric(
                "Canvas",
                f"{w} × {h}"
            )

        with info2:
            st.metric(
                "Character",
                f"{char_w} × {char_h}"
            )

        with info3:
            frame_count = max(
                8,
                int(
                    fps * duration
                )
            )

            st.metric(
                "Frames",
                frame_count
            )

        with info4:
            st.metric(
                "FPS",
                fps
            )

        # ----------------------------------------------------
        # GENERATE
        # ----------------------------------------------------

        generate = st.button(
            "✨ Generate Walk-Away Animation",
            type="primary",
            use_container_width=True
        )

        if generate:

            progress = st.progress(
                0,
                text="Preparing animation..."
            )

            # ------------------------------------------------
            # START / EXIT POSITIONS
            # ------------------------------------------------

            start_x, start_y = (
                get_start_center(
                    bbox,
                    w,
                    h
                )
            )

            exit_x, exit_y = (
                get_exit_center(
                    start_x,
                    start_y,
                    char_w,
                    char_h,
                    w,
                    h,
                    direction
                )
            )

            start = (
                start_x,
                start_y
            )

            exit_pos = (
                exit_x,
                exit_y
            )

            end_scale = (
                end_scale_percent
                /
                100.0
            )

            # ------------------------------------------------
            # RENDER FRAMES
            # ------------------------------------------------

            frames = []

            for i in range(
                frame_count
            ):

                if frame_count <= 1:
                    t = 1.0
                else:
                    t = (
                        i
                        /
                        (
                            frame_count - 1
                        )
                    )

                frame = render_walk_frame(
                    background=background,
                    character=character,
                    alpha=alpha,
                    t=t,
                    start=start,
                    exit_pos=exit_pos,
                    end_scale=end_scale,
                    bob_amount=bob_amount,
                    sway_amount=sway_amount,
                    cycles=cycles,
                    fade_at_end=fade_at_end,
                )

                frames.append(
                    frame
                )

                progress.progress(
                    (i + 1)
                    /
                    frame_count,
                    text=(
                        f"Rendering frame "
                        f"{i + 1}/{frame_count}"
                    )
                )

            progress.empty()

            # ------------------------------------------------
            # CONTACT SHEET
            # ------------------------------------------------

            st.subheader(
                "🎞️ Frame-by-Frame Movement"
            )

            contact = make_contact_sheet(
                frames,
                columns=4
            )

            if contact is not None:

                st.image(
                    bgr_to_rgb(contact),
                    use_container_width=True
                )

            # ------------------------------------------------
            # GIF
            # ------------------------------------------------

            with st.spinner(
                "🎬 Creating GIF..."
            ):

                gif_data = build_gif(
                    frames,
                    fps
                )

            # ------------------------------------------------
            # MP4
            # ------------------------------------------------

            with st.spinner(
                "🎥 Creating MP4..."
            ):

                mp4_data = build_mp4(
                    frames,
                    fps
                )

            # ------------------------------------------------
            # PREVIEW
            # ------------------------------------------------

            st.subheader(
                "🎬 Final Animation"
            )

            if gif_data:

                st.image(
                    gif_data,
                    use_container_width=True
                )

            # ------------------------------------------------
            # DOWNLOADS
            # ------------------------------------------------

            d1, d2 = st.columns(2)

            safe_direction = (
                direction
                .lower()
                .replace(" ", "_")
                .replace("-", "_")
            )

            with d1:

                if gif_data:

                    st.download_button(
                        "⬇️ Download GIF",
                        data=gif_data,
                        file_name=(
                            f"character_walk_"
                            f"{safe_direction}.gif"
                        ),
                        mime="image/gif",
                        use_container_width=True
                    )

            with d2:

                if mp4_data:

                    st.download_button(
                        "🎞️ Download MP4",
                        data=mp4_data,
                        file_name=(
                            f"character_walk_"
                            f"{safe_direction}.mp4"
                        ),
                        mime="video/mp4",
                        use_container_width=True
                    )

            # ------------------------------------------------
            # FINAL INFORMATION
            # ------------------------------------------------

            st.success(
                "✅ Animation generated! "
                "The character progressively leaves the canvas, "
                "while the reconstructed background remains behind."
            )

            st.info(
                "💡 For a stronger 'walking away into the distance' "
                "effect, try Final Character Size = 50–70%. "
                "For simply walking out of the scene, keep it at 100%."
            )

    except Exception as e:

        st.error(
            f"❌ Processing Error: {e}"
        )

        st.exception(e)

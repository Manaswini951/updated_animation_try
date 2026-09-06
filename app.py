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
    page_title="Hand-Drawn Character Walk Away Animator",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🎨 Hand-Drawn Character Walk-Away Animator")

st.markdown(
    """
Turn one hand-drawn scene into a simple frame-by-frame animation.

### Animation concept

**Character enters → stays → walks away → disappears**

The character is separated from the background, while the original
character location is reconstructed so there is no duplicate character
left behind.

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
# BASIC IMAGE HELPERS
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
# BACKGROUND COLOR ESTIMATION
# ============================================================

def get_border_pixels(image, border_size=12):

    h, w = image.shape[:2]

    border_size = max(
        2,
        min(
            border_size,
            h // 4,
            w // 4
        )
    )

    top = image[:border_size]
    bottom = image[h - border_size:h]
    left = image[:, :border_size]
    right = image[:, w - border_size:w]

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

    border = get_border_pixels(image)

    border_lab = cv2.cvtColor(
        border.reshape(-1, 1, 3),
        cv2.COLOR_BGR2LAB
    ).reshape(-1, 3)

    return np.median(
        border_lab,
        axis=0
    ).astype(np.float32)


def estimate_background_gray(image):

    border = get_border_pixels(image)

    border_gray = cv2.cvtColor(
        border.reshape(-1, 1, 3),
        cv2.COLOR_BGR2GRAY
    ).reshape(-1)

    return float(
        np.median(border_gray)
    )


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_map(values):

    values = values.astype(np.float32)

    mn = float(np.min(values))
    mx = float(np.max(values))

    if mx - mn < 1e-6:
        return np.zeros_like(values)

    result = (values - mn) / (mx - mn)

    return np.clip(
        result,
        0.0,
        1.0
    )


# ============================================================
# RAW FOREGROUND SCORE
# ============================================================

def calculate_foreground_score(
    image,
    background_mode="White / Light Paper",
):

    h, w = image.shape[:2]

    # --------------------------------------------------------
    # LAB COLOR DIFFERENCE
    # --------------------------------------------------------

    lab = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2LAB
    )

    bg_lab = estimate_background_lab(
        image
    )

    difference = (
        lab.astype(np.float32)
        -
        bg_lab.reshape(1, 1, 3)
    )

    color_distance = np.sqrt(
        np.sum(
            difference * difference,
            axis=2
        )
    )

    color_distance = normalize_map(
        color_distance
    )

    # --------------------------------------------------------
    # GRAYSCALE
    # --------------------------------------------------------

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    bg_gray = estimate_background_gray(
        image
    )

    if background_mode == "Dark Background":

        darkness = (
            gray.astype(np.float32)
            -
            bg_gray
        )

    else:

        darkness = (
            bg_gray
            -
            gray.astype(np.float32)
        )

    darkness = np.clip(
        darkness,
        0,
        None
    )

    darkness = normalize_map(
        darkness
    )

    # --------------------------------------------------------
    # LOCAL CONTRAST
    # --------------------------------------------------------

    sigma = max(
        5,
        min(h, w) / 80
    )

    local_background = cv2.GaussianBlur(
        gray,
        (0, 0),
        sigmaX=sigma
    )

    if background_mode == "Dark Background":

        local_difference = (
            gray.astype(np.float32)
            -
            local_background.astype(np.float32)
        )

    else:

        local_difference = (
            local_background.astype(np.float32)
            -
            gray.astype(np.float32)
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

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_size, kernel_size)
    )

    blackhat = cv2.morphologyEx(
        gray,
        cv2.MORPH_BLACKHAT,
        kernel
    )

    blackhat = normalize_map(
        blackhat
    )

    # --------------------------------------------------------
    # EDGES
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

    edges = (
        edges.astype(np.float32)
        /
        255.0
    )

    # --------------------------------------------------------
    # COMBINED SCORE
    # --------------------------------------------------------

    score = (
        color_distance * 0.40
        +
        darkness * 0.28
        +
        local_difference * 0.12
        +
        blackhat * 0.15
        +
        edges * 0.05
    )

    score = cv2.GaussianBlur(
        score,
        (5, 5),
        0
    )

    return score


# ============================================================
# DETECT OBJECT COMPONENTS
# ============================================================

def detect_objects(
    image,
    sensitivity=50,
    background_mode="White / Light Paper",
):

    h, w = image.shape[:2]

    score = calculate_foreground_score(
        image,
        background_mode
    )

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

    # Recover strong signals
    color_distance = np.zeros_like(score)

    lab = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2LAB
    )

    bg_lab = estimate_background_lab(
        image
    )

    difference = (
        lab.astype(np.float32)
        -
        bg_lab.reshape(1, 1, 3)
    )

    color_distance = normalize_map(
        np.sqrt(
            np.sum(
                difference * difference,
                axis=2
            )
        )
    )

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    bg_gray = estimate_background_gray(
        image
    )

    if background_mode == "Dark Background":

        darkness = (
            gray.astype(np.float32)
            -
            bg_gray
        )

    else:

        darkness = (
            bg_gray
            -
            gray.astype(np.float32)
        )

    darkness = normalize_map(
        np.clip(
            darkness,
            0,
            None
        )
    )

    strong_signal = (
        (color_distance > 0.20)
        |
        (darkness > 0.20)
        |
        (score > threshold * 0.65)
    )

    mask[
        strong_signal
        &
        (score > threshold * 0.50)
    ] = 255

    # --------------------------------------------------------
    # MORPHOLOGY
    # --------------------------------------------------------

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        np.ones((5, 5), np.uint8),
        iterations=2
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        np.ones((3, 3), np.uint8),
        iterations=1
    )

    # --------------------------------------------------------
    # CONNECTED COMPONENTS
    # --------------------------------------------------------

    num_labels, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            mask,
            connectivity=8
        )
    )

    components = []

    minimum_area = max(
        25,
        int(h * w * 0.000025)
    )

    for label in range(
        1,
        num_labels
    ):

        area = int(
            stats[
                label,
                cv2.CC_STAT_AREA
            ]
        )

        if area < minimum_area:
            continue

        x = int(
            stats[
                label,
                cv2.CC_STAT_LEFT
            ]
        )

        y = int(
            stats[
                label,
                cv2.CC_STAT_TOP
            ]
        )

        cw = int(
            stats[
                label,
                cv2.CC_STAT_WIDTH
            ]
        )

        ch = int(
            stats[
                label,
                cv2.CC_STAT_HEIGHT
            ]
        )

        cx, cy = centroids[label]

        components.append(
            {
                "label": label,
                "area": area,
                "x": x,
                "y": y,
                "w": cw,
                "h": ch,
                "cx": float(cx),
                "cy": float(cy),
            }
        )

    components.sort(
        key=lambda c: c["area"],
        reverse=True
    )

    return mask, labels, components


# ============================================================
# BUILD OBJECT PREVIEW
# ============================================================

def make_object_preview(
    image,
    labels,
    components
):

    preview = image.copy()

    display_components = components[:20]

    for index, comp in enumerate(
        display_components
    ):

        x = comp["x"]
        y = comp["y"]
        w = comp["w"]
        h = comp["h"]

        # Bounding box
        cv2.rectangle(
            preview,
            (x, y),
            (x + w, y + h),
            (0, 0, 255),
            2
        )

        text_x = x + 5
        text_y = max(
            25,
            y + 25
        )

        # Filled label background
        cv2.rectangle(
            preview,
            (
                text_x - 3,
                text_y - 22
            ),
            (
                text_x + 55,
                text_y + 5
            ),
            (255, 255, 255),
            -1
        )

        cv2.putText(
            preview,
            str(index + 1),
            (
                text_x,
                text_y
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
            cv2.LINE_AA
        )

    return preview


# ============================================================
# CREATE SELECTED CHARACTER MASK
# ============================================================

def create_character_mask(
    labels,
    components,
    selected_indices
):

    mask = np.zeros(
        labels.shape,
        dtype=np.uint8
    )

    for index in selected_indices:

        if index < 0:
            continue

        if index >= len(components):
            continue

        label = components[index]["label"]

        mask[
            labels == label
        ] = 255

    return mask


# ============================================================
# IMPROVE CHARACTER MASK
# ============================================================

def refine_character_mask(
    mask,
    dilation=2,
    feather=2
):

    mask = mask.astype(
        np.uint8
    )

    # Connect nearby pieces
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        np.ones((5, 5), np.uint8),
        iterations=2
    )

    # Slightly expand to preserve outlines
    if dilation > 0:

        kernel_size = (
            dilation * 2 + 1
        )

        mask = cv2.dilate(
            mask,
            np.ones(
                (
                    kernel_size,
                    kernel_size
                ),
                np.uint8
            ),
            iterations=1
        )

    # Smooth edges
    if feather > 0:

        kernel_size = (
            feather * 2 + 1
        )

        mask = cv2.GaussianBlur(
            mask,
            (
                kernel_size,
                kernel_size
            ),
            0
        )

    return mask


# ============================================================
# EXTRACT CHARACTER
# ============================================================

def extract_character(
    image,
    mask,
    margin=20
):

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

    character = image[
        y1:y2,
        x1:x2
    ].copy()

    alpha = mask[
        y1:y2,
        x1:x2
    ].copy()

    bbox = (
        x1,
        y1,
        x2,
        y2
    )

    return (
        character,
        alpha,
        bbox
    )


# ============================================================
# RECONSTRUCT EMPTY BACKGROUND
# ============================================================

def reconstruct_background(
    image,
    character_mask,
    inpaint_strength=7,
    extra_margin=8
):

    # IMPORTANT:
    # Only the MAIN CHARACTER gets removed.
    # Other selected background objects are untouched.

    mask = character_mask.copy()

    if extra_margin > 0:

        kernel_size = (
            extra_margin * 2 + 1
        )

        mask = cv2.dilate(
            mask,
            np.ones(
                (
                    kernel_size,
                    kernel_size
                ),
                np.uint8
            ),
            iterations=1
        )

    coverage = (
        np.count_nonzero(mask)
        /
        float(mask.size)
    )

    # Safety check
    if coverage > 0.70:
        return image.copy()

    # First Telea pass
    background = cv2.inpaint(
        image,
        mask,
        inpaint_strength,
        cv2.INPAINT_TELEA
    )

    # Second gentle Navier-Stokes pass
    # helps larger white-paper areas
    try:

        background = cv2.inpaint(
            background,
            mask,
            max(
                3,
                inpaint_strength // 2
            ),
            cv2.INPAINT_NS
        )

    except Exception:
        pass

    return background


# ============================================================
# CHECKERBOARD
# ============================================================

def checkerboard(
    width,
    height,
    square=20
):

    result = np.zeros(
        (
            height,
            width,
            3
        ),
        dtype=np.uint8
    )

    for y in range(
        0,
        height,
        square
    ):

        for x in range(
            0,
            width,
            square
        ):

            if (
                (x // square + y // square)
                % 2
                == 0
            ):

                value = 225

            else:

                value = 245

            result[
                y:min(
                    y + square,
                    height
                ),
                x:min(
                    x + square,
                    width
                )
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
        /
        255.0
    )

    a = a[:, :, None]

    result = (
        character.astype(np.float32)
        *
        a
        +
        bg.astype(np.float32)
        *
        (1 - a)
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
        math.cos(
            math.pi * t
        )
    )


# ============================================================
# POSITION HELPERS
# ============================================================

def direction_vector(
    direction
):

    vectors = {

        "Right": (1, 0),

        "Left": (-1, 0),

        "Down": (0, 1),

        "Up": (0, -1),

        "Diagonal Down-Right": (
            1,
            1
        ),

        "Diagonal Down-Left": (
            -1,
            1
        ),

        "Diagonal Up-Right": (
            1,
            -1
        ),

        "Diagonal Up-Left": (
            -1,
            -1
        ),
    }

    return vectors.get(
        direction,
        (1, 0)
    )


def get_original_center(
    bbox
):

    x1, y1, x2, y2 = bbox

    return (
        (x1 + x2) / 2.0,
        (y1 + y2) / 2.0
    )


def get_outside_position(
    center,
    char_w,
    char_h,
    canvas_w,
    canvas_h,
    direction,
    extra_distance=1.5
):

    cx, cy = center

    margin_x = (
        char_w
        *
        extra_distance
    )

    margin_y = (
        char_h
        *
        extra_distance
    )

    if direction == "Right":

        return (
            canvas_w + margin_x,
            cy
        )

    if direction == "Left":

        return (
            -margin_x,
            cy
        )

    if direction == "Down":

        return (
            cx,
            canvas_h + margin_y
        )

    if direction == "Up":

        return (
            cx,
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
        cy
    )


# ============================================================
# WALKING SEQUENCE
# ============================================================

def sequence_position(
    global_t,
    original_center,
    char_w,
    char_h,
    canvas_w,
    canvas_h,
    direction,
    walk_in_fraction,
    stay_fraction,
    walk_out_fraction,
    bob_amount,
    sway_amount,
    cycles,
    start_scale,
    end_scale,
):

    # --------------------------------------------------------
    # NORMALIZE SECTIONS
    # --------------------------------------------------------

    total = (
        walk_in_fraction
        +
        stay_fraction
        +
        walk_out_fraction
    )

    if total <= 0:
        total = 1.0

    walk_in_end = (
        walk_in_fraction
        /
        total
    )

    stay_end = (
        (
            walk_in_fraction
            +
            stay_fraction
        )
        /
        total
    )

    # --------------------------------------------------------
    # POSITIONS
    # --------------------------------------------------------

    outside = get_outside_position(
        original_center,
        char_w,
        char_h,
        canvas_w,
        canvas_h,
        direction,
        extra_distance=1.7
    )

    # --------------------------------------------------------
    # WALK IN
    # --------------------------------------------------------

    if global_t < walk_in_end:

        local_t = (
            global_t
            /
            max(
                walk_in_end,
                1e-6
            )
        )

        movement = ease_in_out(
            local_t
        )

        x = (
            outside[0]
            +
            (
                original_center[0]
                -
                outside[0]
            )
            *
            movement
        )

        y = (
            outside[1]
            +
            (
                original_center[1]
                -
                outside[1]
            )
            *
            movement
        )

        phase = (
            local_t
            *
            cycles
            *
            math.pi
            *
            2
        )

        bob = (
            math.sin(phase)
            *
            bob_amount
        )

        sway = (
            math.sin(
                phase
                +
                math.pi / 2
            )
            *
            sway_amount
        )

        scale = start_scale

        return (
            x,
            y + bob,
            sway,
            scale
        )

    # --------------------------------------------------------
    # STAY
    # --------------------------------------------------------

    if global_t < stay_end:

        local_t = (
            (
                global_t
                -
                walk_in_end
            )
            /
            max(
                stay_end
                -
                walk_in_end,
                1e-6
            )
        )

        # Very subtle breathing while standing
        breathing = (
            math.sin(
                local_t
                *
                math.pi
                *
                2
            )
            *
            min(
                bob_amount * 0.18,
                3
            )
        )

        return (
            original_center[0],
            original_center[1] + breathing,
            0.0,
            start_scale
        )

    # --------------------------------------------------------
    # WALK OUT
    # --------------------------------------------------------

    local_t = (
        (
            global_t
            -
            stay_end
        )
        /
        max(
            1.0
            -
            stay_end,
            1e-6
        )
    )

    movement = ease_in_out(
        local_t
    )

    x = (
        original_center[0]
        +
        (
            outside[0]
            -
            original_center[0]
        )
        *
        movement
    )

    y = (
        original_center[1]
        +
        (
            outside[1]
            -
            original_center[1]
        )
        *
        movement
    )

    phase = (
        local_t
        *
        cycles
        *
        math.pi
        *
        2
    )

    # Less bobbing near the exit
    bob_fade = (
        1.0
        -
        smoothstep(
            max(
                0.0,
                (
                    local_t
                    -
                    0.70
                )
                /
                0.30
            )
        )
    )

    bob = (
        math.sin(phase)
        *
        bob_amount
        *
        bob_fade
    )

    sway = (
        math.sin(
            phase
            +
            math.pi / 2
        )
        *
        sway_amount
        *
        bob_fade
    )

    scale = (
        start_scale
        +
        (
            end_scale
            -
            start_scale
        )
        *
        ease_in_out(local_t)
    )

    return (
        x,
        y + bob,
        sway,
        scale
    )


# ============================================================
# TRANSFORM CHARACTER
# ============================================================

def transform_character(
    character,
    alpha,
    scale,
    angle
):

    h, w = character.shape[:2]

    scale = max(
        0.05,
        float(scale)
    )

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
        (
            new_w,
            new_h
        ),
        interpolation=cv2.INTER_LINEAR
    )

    resized_alpha = cv2.resize(
        alpha,
        (
            new_w,
            new_h
        ),
        interpolation=cv2.INTER_LINEAR
    )

    center = (
        new_w / 2.0,
        new_h / 2.0
    )

    matrix = cv2.getRotationMatrix2D(
        center,
        angle,
        1.0
    )

    cos = abs(
        matrix[0, 0]
    )

    sin = abs(
        matrix[0, 1]
    )

    bound_w = max(
        2,
        int(
            new_h * sin
            +
            new_w * cos
        )
    )

    bound_h = max(
        2,
        int(
            new_h * cos
            +
            new_w * sin
        )
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

    transformed = cv2.warpAffine(
        resized,
        matrix,
        (
            bound_w,
            bound_h
        ),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255)
    )

    transformed_alpha = cv2.warpAffine(
        resized_alpha,
        matrix,
        (
            bound_w,
            bound_h
        ),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )

    return (
        transformed,
        transformed_alpha
    )


# ============================================================
# COMPOSITE CHARACTER
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

    char_h, char_w = character.shape[:2]

    x1 = int(
        round(
            center_x
            -
            char_w / 2
        )
    )

    y1 = int(
        round(
            center_y
            -
            char_h / 2
        )
    )

    x2 = x1 + char_w
    y2 = y1 + char_h

    canvas_h, canvas_w = (
        canvas.shape[:2]
    )

    # Completely outside
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

    # Clip
    cx1 = max(
        0,
        x1
    )

    cy1 = max(
        0,
        y1
    )

    cx2 = min(
        canvas_w,
        x2
    )

    cy2 = min(
        canvas_h,
        y2
    )

    src_x1 = cx1 - x1
    src_y1 = cy1 - y1

    src_x2 = (
        src_x1
        +
        (
            cx2 - cx1
        )
    )

    src_y2 = (
        src_y1
        +
        (
            cy2 - cy1
        )
    )

    char_crop = character[
        src_y1:src_y2,
        src_x1:src_x2
    ]

    alpha_crop = alpha[
        src_y1:src_y2,
        src_x1:src_x2
    ]

    a = (
        alpha_crop.astype(
            np.float32
        )
        /
        255.0
    )

    a *= float(
        np.clip(
            fade,
            0.0,
            1.0
        )
    )

    a = a[:, :, None]

    bg_crop = canvas[
        cy1:cy2,
        cx1:cx2
    ].astype(
        np.float32
    )

    result = (
        char_crop.astype(
            np.float32
        )
        *
        a
        +
        bg_crop
        *
        (1.0 - a)
    )

    canvas[
        cy1:cy2,
        cx1:cx2
    ] = np.clip(
        result,
        0,
        255
    ).astype(
        np.uint8
    )

    return canvas


# ============================================================
# RENDER FRAME
# ============================================================

def render_frame(
    background,
    character,
    alpha,
    global_t,
    original_center,
    char_w,
    char_h,
    canvas_w,
    canvas_h,
    direction,
    walk_in_fraction,
    stay_fraction,
    walk_out_fraction,
    bob_amount,
    sway_amount,
    cycles,
    start_scale,
    end_scale,
    fade_at_end,
):

    (
        x,
        y,
        angle,
        scale
    ) = sequence_position(
        global_t,
        original_center,
        char_w,
        char_h,
        canvas_w,
        canvas_h,
        direction,
        walk_in_fraction,
        stay_fraction,
        walk_out_fraction,
        bob_amount,
        sway_amount,
        cycles,
        start_scale,
        end_scale
    )

    transformed_character, transformed_alpha = (
        transform_character(
            character,
            alpha,
            scale,
            angle
        )
    )

    # Optional fade ONLY near final exit.
    fade = 1.0

    if fade_at_end and global_t > 0.92:

        fade = (
            1.0
            -
            (
                global_t
                -
                0.92
            )
            /
            0.08
        )

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
    thumb_width=260,
    max_width=1000
):

    if not frames:
        return None

    thumbs = []

    for frame in frames:

        h, w = frame.shape[:2]

        thumb_h = max(
            1,
            int(
                h
                *
                thumb_width
                /
                max(1, w)
            )
        )

        thumb = cv2.resize(
            frame,
            (
                thumb_width,
                thumb_h
            ),
            interpolation=cv2.INTER_AREA
        )

        thumbs.append(
            thumb
        )

    rows = math.ceil(
        len(thumbs)
        /
        columns
    )

    cell_h = max(
        t.shape[0]
        for t in thumbs
    )

    cell_w = max(
        t.shape[1]
        for t in thumbs
    )

    sheet = np.ones(
        (
            rows * cell_h,
            columns * cell_w,
            3
        ),
        dtype=np.uint8
    ) * 255

    for i, thumb in enumerate(
        thumbs
    ):

        row = i // columns
        col = i % columns

        x = (
            col
            *
            cell_w
        )

        y = (
            row
            *
            cell_h
        )

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
                int(
                    sheet.shape[1]
                    *
                    scale
                ),
                int(
                    sheet.shape[0]
                    *
                    scale
                )
            ),
            interpolation=cv2.INTER_AREA
        )

    return sheet


# ============================================================
# GIF
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
        int(
            1000 / fps
        )
    )

    for frame in frames:

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        pil = Image.fromarray(
            rgb
        ).convert(
            "P",
            palette=Image.ADAPTIVE
        )

        prepared.append(
            pil
        )

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
# MP4
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
            (
                w,
                h
            )
        )

        if not writer.isOpened():
            return None

        for frame in frames:

            writer.write(
                frame
            )

        writer.release()

        with open(
            temp_path,
            "rb"
        ) as f:

            return f.read()

    except Exception:

        return None

    finally:

        if os.path.exists(
            temp_path
        ):

            try:
                os.remove(
                    temp_path
                )
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
        24
    ],
    value=12
)

duration = st.sidebar.slider(
    "Total Animation Duration",
    min_value=2.0,
    max_value=10.0,
    value=5.0,
    step=0.5
)

st.sidebar.markdown("---")

st.sidebar.header(
    "🚶 Character Timing"
)

walk_in_percent = st.sidebar.slider(
    "Walk In",
    min_value=0,
    max_value=50,
    value=25,
    step=5
)

stay_percent = st.sidebar.slider(
    "Stay In Scene",
    min_value=0,
    max_value=70,
    value=30,
    step=5
)

walk_out_percent = st.sidebar.slider(
    "Walk Out",
    min_value=10,
    max_value=70,
    value=45,
    step=5
)

st.sidebar.caption(
    "The percentages are automatically normalized. "
    "For example: 25% walk in → 30% stay → 45% walk out."
)

st.sidebar.markdown("---")

st.sidebar.header(
    "🚶 Walking Motion"
)

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
    max_value=15,
    value=6
)

st.sidebar.markdown("---")

st.sidebar.header(
    "📏 Walking Away"
)

end_scale_percent = st.sidebar.slider(
    "Final Character Size",
    min_value=20,
    max_value=100,
    value=100,
    step=5
)

st.sidebar.caption(
    "100% = character keeps the same size. "
    "Smaller values make the character appear to walk into the distance."
)

fade_at_end = st.sidebar.checkbox(
    "Fade at very end",
    value=False
)

st.sidebar.markdown("---")

st.sidebar.header(
    "🎨 Character Extraction"
)

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
    "Higher values select less background. "
    "Lower values help recover faint parts of the character."
)


# ============================================================
# FILE UPLOAD
# ============================================================

uploaded = st.file_uploader(
    "Upload your hand-drawn scene",
    type=[
        "jpg",
        "jpeg",
        "png",
        "webp"
    ]
)


# ============================================================
# MAIN APPLICATION
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
        # DETECT OBJECTS
        # ----------------------------------------------------

        with st.spinner(
            "🔍 Detecting objects in the drawing..."
        ):

            raw_mask, labels, components = (
                detect_objects(
                    image,
                    sensitivity=sensitivity,
                    background_mode=background_mode
                )
            )

        if not components:

            st.error(
                "❌ No foreground objects could be detected."
            )

            st.info(
                "Try lowering Foreground Sensitivity."
            )

            st.image(
                bgr_to_rgb(image),
                use_container_width=True
            )

            st.stop()

        # ----------------------------------------------------
        # OBJECT PREVIEW
        # ----------------------------------------------------

        st.subheader(
            "🎯 Choose the Main Character"
        )

        st.markdown(
            """
The numbered boxes below are detected foreground objects.

**Select ONLY the objects that belong to the character.**

Everything else will remain in the scene as a stationary
background object.
"""
        )

        object_preview = make_object_preview(
            image,
            labels,
            components
        )

        st.image(
            bgr_to_rgb(object_preview),
            use_container_width=True
        )

        # ----------------------------------------------------
        # OBJECT SELECTOR
        # ----------------------------------------------------

        max_objects = min(
            len(components),
            20
        )

        object_options = []

        for i in range(
            max_objects
        ):

            comp = components[i]

            object_options.append(
                (
                    f"Object {i + 1} "
                    f"— area {comp['area']} "
                    f"— {comp['w']}×{comp['h']}"
                )
            )

        # Automatically suggest largest central object(s)
        center_x = w / 2.0
        center_y = h / 2.0

        suggested = []

        for i, comp in enumerate(
            components[:max_objects]
        ):

            distance = math.sqrt(
                (
                    comp["cx"]
                    -
                    center_x
                ) ** 2
                +
                (
                    comp["cy"]
                    -
                    center_y
                ) ** 2
            )

            normalized_distance = (
                distance
                /
                max(
                    1,
                    math.sqrt(
                        w * w
                        +
                        h * h
                    )
                )
            )

            area_ratio = (
                comp["area"]
                /
                max(
                    1,
                    components[0]["area"]
                )
            )

            if (
                i == 0
                or
                (
                    area_ratio > 0.12
                    and
                    normalized_distance < 0.40
                )
            ):

                suggested.append(
                    object_options[i]
                )

        selected_object_labels = st.multiselect(
            "Main Character Objects",
            object_options,
            default=suggested,
            help=(
                "Select all detected pieces that belong "
                "to the character. Trees, chairs, balls, "
                "houses, etc. that are not selected will "
                "stay in the background."
            )
        )

        selected_indices = [
            object_options.index(label)
            for label in selected_object_labels
        ]

        # ----------------------------------------------------
        # WARNING
        # ----------------------------------------------------

        if not selected_indices:

            st.warning(
                "⚠️ Select at least one object as the main character."
            )

            st.stop()

        # ----------------------------------------------------
        # BUILD CHARACTER MASK
        # ----------------------------------------------------

        character_mask = create_character_mask(
            labels,
            components,
            selected_indices
        )

        # Slightly connect selected character pieces
        character_mask = refine_character_mask(
            character_mask,
            dilation=2,
            feather=2
        )

        # ----------------------------------------------------
        # EXTRACT CHARACTER
        # ----------------------------------------------------

        with st.spinner(
            "✂️ Extracting complete character..."
        ):

            character, alpha, bbox = (
                extract_character(
                    image,
                    character_mask,
                    margin=20
                )
            )

        if character is None:

            st.error(
                "❌ Could not extract the selected character."
            )

            st.stop()

        char_h, char_w = character.shape[:2]

        # ----------------------------------------------------
        # RECONSTRUCT BACKGROUND
        # ----------------------------------------------------

        with st.spinner(
            "🧹 Removing the character from the original scene..."
        ):

            background = reconstruct_background(
                image,
                character_mask,
                inpaint_strength=7,
                extra_margin=8
            )

        # ----------------------------------------------------
        # PREVIEWS
        # ----------------------------------------------------

        st.markdown("---")

        st.subheader(
            "🔍 Scene Separation"
        )

        c1, c2, c3 = st.columns(3)

        with c1:

            st.markdown(
                "**Original Scene**"
            )

            st.image(
                bgr_to_rgb(image),
                use_container_width=True
            )

        with c2:

            st.markdown(
                "**Extracted Main Character**"
            )

            character_preview = (
                composite_character_preview(
                    character,
                    alpha
                )
            )

            st.image(
                bgr_to_rgb(
                    character_preview
                ),
                use_container_width=True
            )

        with c3:

            st.markdown(
                "**Character Removed / Clean Background**"
            )

            st.image(
                bgr_to_rgb(background),
                use_container_width=True
            )

        # ----------------------------------------------------
        # IMPORTANT INFORMATION
        # ----------------------------------------------------

        st.info(
            "🧠 The animation uses the CLEAN BACKGROUND as its base. "
            "The extracted character is then placed on top of it "
            "frame-by-frame. This prevents the original character "
            "from remaining behind when it walks away."
        )

        # ----------------------------------------------------
        # METRICS
        # ----------------------------------------------------

        st.markdown("---")

        frame_count = max(
            8,
            int(
                fps * duration
            )
        )

        m1, m2, m3, m4 = st.columns(4)

        with m1:

            st.metric(
                "Canvas",
                f"{w} × {h}"
            )

        with m2:

            st.metric(
                "Character",
                f"{char_w} × {char_h}"
            )

        with m3:

            st.metric(
                "Frames",
                frame_count
            )

        with m4:

            st.metric(
                "FPS",
                fps
            )

        # ----------------------------------------------------
        # GENERATE
        # ----------------------------------------------------

        generate = st.button(
            "✨ Generate Walk-In → Stay → Walk-Out",
            type="primary",
            use_container_width=True
        )

        if generate:

            progress = st.progress(
                0,
                text="Preparing animation..."
            )

            # ------------------------------------------------
            # ORIGINAL CHARACTER CENTER
            # ------------------------------------------------

            original_center = (
                get_original_center(
                    bbox
                )
            )

            start_scale = 1.0

            end_scale = (
                end_scale_percent
                /
                100.0
            )

            # Normalize timing
            total_timing = (
                walk_in_percent
                +
                stay_percent
                +
                walk_out_percent
            )

            if total_timing <= 0:
                total_timing = 100.0

            walk_in_fraction = (
                walk_in_percent
                /
                total_timing
            )

            stay_fraction = (
                stay_percent
                /
                total_timing
            )

            walk_out_fraction = (
                walk_out_percent
                /
                total_timing
            )

            # ------------------------------------------------
            # RENDER
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
                            frame_count
                            -
                            1
                        )
                    )

                frame = render_frame(
                    background=background,
                    character=character,
                    alpha=alpha,
                    global_t=t,
                    original_center=original_center,
                    char_w=char_w,
                    char_h=char_h,
                    canvas_w=w,
                    canvas_h=h,
                    direction=direction,
                    walk_in_fraction=walk_in_fraction,
                    stay_fraction=stay_fraction,
                    walk_out_fraction=walk_out_fraction,
                    bob_amount=bob_amount,
                    sway_amount=sway_amount,
                    cycles=cycles,
                    start_scale=start_scale,
                    end_scale=end_scale,
                    fade_at_end=fade_at_end,
                )

                frames.append(
                    frame
                )

                progress.progress(
                    (
                        i + 1
                    )
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
                "🎞️ Frame-by-Frame Animation"
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
                .replace(
                    " ",
                    "_"
                )
                .replace(
                    "-",
                    "_"
                )
            )

            with d1:

                if gif_data:

                    st.download_button(
                        "⬇️ Download GIF",
                        data=gif_data,
                        file_name=(
                            "character_"
                            "walk_in_stay_"
                            f"walk_out_"
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
                            "character_"
                            "walk_in_stay_"
                            "walk_out_"
                            f"{safe_direction}.mp4"
                        ),
                        mime="video/mp4",
                        use_container_width=True
                    )

            # ------------------------------------------------
            # SUCCESS
            # ------------------------------------------------

            st.success(
                "✅ Animation generated! "
                "The character walks into the scene, stays, "
                "and then walks completely away while the "
                "original character location remains a "
                "reconstructed background."
            )

            st.markdown(
                """
### 💡 Recommended settings

For a natural little story animation:

- **Walk In:** 20–30%
- **Stay:** 25–35%
- **Walk Out:** 40–50%
- **FPS:** 12–15
- **Walking Steps:** 5–7
- **Final Character Size:** 100% for simply leaving the scene
- **Final Character Size:** 50–70% for a stronger "walking into distance" effect

If your character should simply enter, pause, and then leave,
keep the final size at **100%**.
"""
            )

    except Exception as e:

        st.error(
            f"❌ Processing Error: {e}"
        )

        st.exception(e)

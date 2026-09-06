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
    page_title="Hand-Drawn Frame Animator Pro",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🎨 Hand-Drawn Frame Animator Pro")

st.markdown(
    """
Turn a single hand-drawn character into a frame-by-frame style animation.

The engine extracts the original artwork, separates colored parts,
creates many slightly different poses, and plays those frames rapidly
to create a traditional hand-drawn animation feel.
"""
)


# ============================================================
# CONSTANTS
# ============================================================

MAX_IMAGE_SIZE = 1100

MOTIONS = [
    "Idle Breathing",
    "Wave",
    "Walk",
    "Run",
    "Jump",
    "Dance",
    "Bounce",
    "Shake",
    "Float",
    "Celebrate",
    "Move Across Canvas",
    "Exit Right",
    "Exit Left",
]


# ============================================================
# COLOR UTILITIES
# ============================================================

def get_color_name(rgb):
    r, g, b = [int(x) for x in rgb]

    pixel = np.uint8([[[b, g, r]]])
    hsv = cv2.cvtColor(pixel, cv2.COLOR_BGR2HSV)[0][0]

    hue = int(hsv[0])
    sat = int(hsv[1])

    if sat < 35:
        return "Neutral"

    if hue < 8 or hue >= 172:
        return "Red"
    elif hue < 22:
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


def extract_dominant_colors(image, num_clusters=8):
    """
    Detect dominant saturated colors.

    The image is resized before clustering so this remains reasonably
    fast on Streamlit Cloud.
    """

    try:
        small = cv2.resize(
            image,
            (160, 160),
            interpolation=cv2.INTER_AREA,
        )

        hsv_small = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)

        saturation = hsv_small[:, :, 1]

        valid = saturation > 45

        pixels = small[valid].reshape(-1, 3)

        if len(pixels) < 50:
            return []

        pixels = pixels.astype(np.float32)

        k = min(
            num_clusters,
            max(2, len(pixels) // 80),
        )

        criteria = (
            cv2.TERM_CRITERIA_EPS +
            cv2.TERM_CRITERIA_MAX_ITER,
            30,
            1.0,
        )

        _, labels, centers = cv2.kmeans(
            pixels,
            k,
            None,
            criteria,
            5,
            cv2.KMEANS_PP_CENTERS,
        )

        counts = np.bincount(
            labels.flatten(),
            minlength=k,
        )

        total = max(1, int(np.sum(counts)))

        detected = []

        for idx, center in enumerate(centers):

            b, g, r = [int(x) for x in center]

            hsv = cv2.cvtColor(
                np.uint8([[[b, g, r]]]),
                cv2.COLOR_BGR2HSV,
            )[0][0]

            h_val = int(hsv[0])
            s_val = int(hsv[1])
            v_val = int(hsv[2])

            coverage = counts[idx] / total * 100

            if coverage < 2.0:
                continue

            if s_val < 45:
                continue

            hex_code = f"#{r:02x}{g:02x}{b:02x}"

            detected.append(
                {
                    "label": (
                        f"{get_color_name((r, g, b))} "
                        f"({hex_code}) "
                        f"{coverage:.1f}%"
                    ),
                    "hsv": (
                        h_val,
                        s_val,
                        v_val,
                    ),
                    "hex": hex_code,
                    "rgb": (
                        r,
                        g,
                        b,
                    ),
                    "coverage": coverage,
                }
            )

        detected.sort(
            key=lambda x: x["coverage"],
            reverse=True,
        )

        return detected

    except Exception:
        return []


# ============================================================
# COLOR MASKING
# ============================================================

def make_color_mask(image, color):
    """
    Build a reasonably tolerant HSV mask around a detected color.
    """

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV,
    )

    h0, s0, v0 = color["hsv"]

    hue_width = 18

    sat_low = max(30, s0 - 90)
    val_low = max(20, v0 - 100)

    if h0 - hue_width < 0:

        lower1 = np.array(
            [
                0,
                sat_low,
                val_low,
            ],
            dtype=np.uint8,
        )

        upper1 = np.array(
            [
                h0 + hue_width,
                255,
                255,
            ],
            dtype=np.uint8,
        )

        lower2 = np.array(
            [
                180 + h0 - hue_width,
                sat_low,
                val_low,
            ],
            dtype=np.uint8,
        )

        upper2 = np.array(
            [
                179,
                255,
                255,
            ],
            dtype=np.uint8,
        )

        mask = cv2.bitwise_or(
            cv2.inRange(hsv, lower1, upper1),
            cv2.inRange(hsv, lower2, upper2),
        )

    elif h0 + hue_width > 179:

        lower1 = np.array(
            [
                h0 - hue_width,
                sat_low,
                val_low,
            ],
            dtype=np.uint8,
        )

        upper1 = np.array(
            [
                179,
                255,
                255,
            ],
            dtype=np.uint8,
        )

        lower2 = np.array(
            [
                0,
                sat_low,
                val_low,
            ],
            dtype=np.uint8,
        )

        upper2 = np.array(
            [
                (h0 + hue_width) - 180,
                255,
                255,
            ],
            dtype=np.uint8,
        )

        mask = cv2.bitwise_or(
            cv2.inRange(hsv, lower1, upper1),
            cv2.inRange(hsv, lower2, upper2),
        )

    else:

        lower = np.array(
            [
                max(0, h0 - hue_width),
                sat_low,
                val_low,
            ],
            dtype=np.uint8,
        )

        upper = np.array(
            [
                min(179, h0 + hue_width),
                255,
                255,
            ],
            dtype=np.uint8,
        )

        mask = cv2.inRange(
            hsv,
            lower,
            upper,
        )

    kernel_small = np.ones(
        (3, 3),
        np.uint8,
    )

    kernel_medium = np.ones(
        (5, 5),
        np.uint8,
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel_small,
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel_medium,
    )

    return mask


# ============================================================
# GEOMETRY UTILITIES
# ============================================================

def component_boundary(mask):

    if mask.dtype == bool:
        mask = mask.astype(np.uint8) * 255

    elif mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
    )

    if not contours:
        return np.empty(
            (0, 2),
            dtype=np.float32,
        )

    points = np.vstack(
        [
            c.reshape(-1, 2)
            for c in contours
        ]
    )

    return points.astype(
        np.float32
    )


def farthest_point(points, origin):

    if len(points) == 0:
        return origin

    origin = np.array(
        origin,
        dtype=np.float32,
    )

    diff = points - origin

    dist = np.sum(
        diff * diff,
        axis=1,
    )

    return tuple(
        points[np.argmax(dist)]
    )


def closest_points_between_masks(mask_a, mask_b):

    pts_a = component_boundary(mask_a)
    pts_b = component_boundary(mask_b)

    if (
        len(pts_a) == 0
        or len(pts_b) == 0
    ):
        return (
            None,
            None,
            float("inf"),
        )

    max_points = 350

    if len(pts_a) > max_points:
        indices = np.linspace(
            0,
            len(pts_a) - 1,
            max_points,
        ).astype(int)

        pts_a = pts_a[indices]

    if len(pts_b) > max_points:
        indices = np.linspace(
            0,
            len(pts_b) - 1,
            max_points,
        ).astype(int)

        pts_b = pts_b[indices]

    diff = (
        pts_a[:, None, :]
        -
        pts_b[None, :, :]
    )

    dist2 = np.sum(
        diff * diff,
        axis=2,
    )

    ia, ib = np.unravel_index(
        np.argmin(dist2),
        dist2.shape,
    )

    return (
        tuple(pts_a[ia]),
        tuple(pts_b[ib]),
        float(
            math.sqrt(
                dist2[ia, ib]
            )
        ),
    )


# ============================================================
# PART EXTRACTION
# ============================================================

def detect_parts(
    image,
    selected_colors,
):
    """
    Detect connected colored components.

    Each component becomes an independent animation part.
    """

    h, w = image.shape[:2]

    all_parts = []

    for color_index, color in enumerate(
        selected_colors
    ):

        mask = make_color_mask(
            image,
            color,
        )

        num_labels, labels, stats, centroids = (
            cv2.connectedComponentsWithStats(
                mask,
                connectivity=8,
            )
        )

        min_area = max(
            180,
            int(
                h * w * 0.00018
            ),
        )

        for i in range(
            1,
            num_labels,
        ):

            area = int(
                stats[
                    i,
                    cv2.CC_STAT_AREA
                ]
            )

            if area < min_area:
                continue

            component_mask = (
                labels == i
            ).astype(np.uint8) * 255

            ys, xs = np.where(
                component_mask > 0
            )

            if len(xs) < 20:
                continue

            x, y, cw, ch = [
                int(v)
                for v in stats[i, :4]
            ]

            cx, cy = centroids[i]

            boundary = component_boundary(
                component_mask
            )

            if len(boundary) == 0:
                continue

            pts = boundary.astype(
                np.float32
            )

            if len(pts) > 1000:
                indices = np.linspace(
                    0,
                    len(pts) - 1,
                    1000,
                ).astype(int)

                pts_for_pca = pts[indices]

            else:
                pts_for_pca = pts

            centered = (
                pts_for_pca
                -
                np.mean(
                    pts_for_pca,
                    axis=0,
                )
            )

            if len(centered) > 2:

                covariance = np.cov(
                    centered.T
                )

                eigenvalues, eigenvectors = (
                    np.linalg.eigh(
                        covariance
                    )
                )

                axis = eigenvectors[
                    :,
                    np.argmax(
                        eigenvalues
                    ),
                ]

                projections = centered @ axis

                p_min = tuple(
                    pts_for_pca[
                        np.argmin(
                            projections
                        )
                    ]
                )

                p_max = tuple(
                    pts_for_pca[
                        np.argmax(
                            projections
                        )
                    ]
                )

            else:

                p_min = (
                    float(cx),
                    float(cy),
                )

                p_max = (
                    float(cx),
                    float(cy),
                )

            length = float(
                np.linalg.norm(
                    np.array(p_max)
                    -
                    np.array(p_min)
                )
            )

            all_parts.append(
                {
                    "id": len(all_parts),
                    "color_index": color_index,
                    "color_name": color["label"],
                    "mask": component_mask > 0,
                    "area": area,
                    "center": (
                        float(cx),
                        float(cy),
                    ),
                    "bbox": (
                        x,
                        y,
                        cw,
                        ch,
                    ),
                    "base": (
                        float(cx),
                        float(cy),
                    ),
                    "tip": p_max,
                    "parent": None,
                    "depth": 0,
                    "length": length,
                }
            )

    if not all_parts:
        return []

    # --------------------------------------------------------
    # HIERARCHY DETECTION
    # --------------------------------------------------------

    for part in all_parts:

        best_parent = None
        best_distance = float("inf")
        best_p1 = None

        for candidate in all_parts:

            if (
                candidate["id"]
                ==
                part["id"]
            ):
                continue

            if (
                candidate["area"]
                <
                part["area"] * 0.30
            ):
                continue

            p1, p2, distance = (
                closest_points_between_masks(
                    part["mask"],
                    candidate["mask"],
                )
            )

            if p1 is None:
                continue

            tolerance = max(
                70,
                min(w, h) * 0.12,
            )

            if (
                distance < tolerance
                and distance < best_distance
            ):

                best_distance = distance
                best_parent = candidate
                best_p1 = p1

        if best_parent is not None:

            part["parent"] = (
                best_parent["id"]
            )

            part["base"] = (
                float(best_p1[0]),
                float(best_p1[1]),
            )

            child_boundary = (
                component_boundary(
                    part["mask"].astype(
                        np.uint8
                    ) * 255
                )
            )

            part["tip"] = farthest_point(
                child_boundary,
                part["base"],
            )

        else:

            part["parent"] = None

            part["base"] = (
                part["center"]
            )

            part["tip"] = farthest_point(
                component_boundary(
                    part["mask"].astype(
                        np.uint8
                    ) * 255
                ),
                part["center"],
            )

    # --------------------------------------------------------
    # DEPTH
    # --------------------------------------------------------

    def get_depth(part):

        visited = set()

        current = part
        depth = 0

        while (
            current["parent"]
            is not None
            and depth < 20
        ):

            if current["id"] in visited:
                break

            visited.add(
                current["id"]
            )

            parent = next(
                (
                    p
                    for p in all_parts
                    if p["id"]
                    ==
                    current["parent"]
                ),
                None,
            )

            if parent is None:
                break

            current = parent
            depth += 1

        return depth

    for part in all_parts:
        part["depth"] = get_depth(
            part
        )

    # Parent first
    all_parts.sort(
        key=lambda p: p["depth"]
    )

    return all_parts


# ============================================================
# BACKGROUND EXTRACTION
# ============================================================

def prepare_background(
    original,
    parts,
):
    """
    Remove movable parts from the original image and
    reconstruct the background.
    """

    if not parts:
        return original.copy()

    combined = np.zeros(
        original.shape[:2],
        dtype=np.uint8,
    )

    for part in parts:

        mask = (
            part["mask"]
            .astype(np.uint8)
            * 255
        )

        # Slightly enlarge the removed area so that
        # old pixels do not remain around moving objects.
        mask = cv2.dilate(
            mask,
            np.ones(
                (5, 5),
                np.uint8,
            ),
            iterations=2,
        )

        combined = cv2.bitwise_or(
            combined,
            mask,
        )

    coverage = (
        np.count_nonzero(combined)
        /
        float(combined.size)
    )

    # If nearly the whole image was selected,
    # don't destroy the original background.
    if coverage > 0.55:
        return original.copy()

    try:

        background = cv2.inpaint(
            original,
            combined,
            7,
            cv2.INPAINT_TELEA,
        )

        return background

    except Exception:

        return original.copy()


# ============================================================
# AFFINE TRANSFORM
# ============================================================

def get_affine_matrix(
    angle_deg,
    tx,
    ty,
    pivot,
):
    """
    Rotation around a specific pivot plus translation.
    """

    rad = math.radians(
        angle_deg
    )

    cos_a = math.cos(rad)
    sin_a = math.sin(rad)

    px, py = pivot

    M = np.array(
        [
            [
                cos_a,
                -sin_a,
                px
                -
                px * cos_a
                +
                py * sin_a
                +
                tx,
            ],
            [
                sin_a,
                cos_a,
                py
                -
                px * sin_a
                -
                py * cos_a
                +
                ty,
            ],
            [
                0,
                0,
                1,
            ],
        ],
        dtype=np.float32,
    )

    return M


# ============================================================
# EASING
# ============================================================

def smoothstep(t):
    """
    Smooth acceleration/deceleration.
    """

    t = max(
        0.0,
        min(1.0, t),
    )

    return (
        t
        *
        t
        *
        (
            3.0
            -
            2.0 * t
        )
    )


def sine_ease(t):
    """
    Smooth looping motion.
    """

    return (
        0.5
        -
        0.5
        *
        math.cos(
            2.0
            *
            math.pi
            *
            t
        )
    )


# ============================================================
# MOTION ENGINE
# ============================================================

def motion_for_part(
    motion,
    part,
    normalized_time,
    intensity,
    canvas_width,
    canvas_height,
):
    """
    Calculates tiny per-frame movement.

    The important difference from the original version is that
    the transformation is applied to the extracted artwork itself,
    rather than repeatedly transforming the entire source image.
    """

    t = max(
        0.0,
        min(
            0.999999,
            normalized_time,
        ),
    )

    phase = (
        2.0
        *
        math.pi
        *
        t
    )

    sine = math.sin(phase)
    cosine = math.cos(phase)

    side = (
        1
        if part["id"] % 2 == 0
        else -1
    )

    depth = part.get(
        "depth",
        0,
    )

    # Lower values prevent very small parts from
    # flying around too much.
    strength = (
        intensity
        *
        (
            0.55
            +
            min(depth, 3)
            *
            0.12
        )
    )

    angle = 0.0
    tx = 0.0
    ty = 0.0

    # --------------------------------------------------------
    # IDLE BREATHING
    # --------------------------------------------------------

    if motion == "Idle Breathing":

        body_factor = (
            0.18
            if depth == 0
            else 0.35
        )

        angle = (
            sine
            *
            strength
            *
            body_factor
        )

        if depth == 0:
            ty = (
                sine
                *
                strength
                *
                0.12
            )

    # --------------------------------------------------------
    # WAVE
    # --------------------------------------------------------

    elif motion == "Wave":

        wave = math.sin(
            2
            *
            math.pi
            *
            t
            +
            depth
            *
            0.30
        )

        angle = (
            wave
            *
            strength
            *
            (
                1.2
                if depth >= 1
                else 0.25
            )
        )

        tx = (
            wave
            *
            strength
            *
            0.08
        )

    # --------------------------------------------------------
    # WALK
    # --------------------------------------------------------

    elif motion == "Walk":

        walk_phase = (
            2
            *
            math.pi
            *
            t
            +
            (
                math.pi
                if side < 0
                else 0
            )
        )

        if depth > 0:

            angle = (
                math.sin(
                    walk_phase
                )
                *
                strength
                *
                1.15
            )

            ty = (
                max(
                    0,
                    -math.cos(
                        walk_phase
                    ),
                )
                *
                strength
                *
                0.15
            )

        else:

            angle = (
                sine
                *
                strength
                *
                0.10
            )

            ty = (
                abs(sine)
                *
                strength
                *
                0.12
            )

    # --------------------------------------------------------
    # RUN
    # --------------------------------------------------------

    elif motion == "Run":

        run_phase = (
            4
            *
            math.pi
            *
            t
            +
            (
                math.pi
                if side < 0
                else 0
            )
        )

        if depth > 0:

            angle = (
                math.sin(
                    run_phase
                )
                *
                strength
                *
                1.7
            )

        else:

            angle = (
                sine
                *
                strength
                *
                0.22
            )

        ty = (
            abs(
                math.sin(
                    4
                    *
                    math.pi
                    *
                    t
                )
            )
            *
            strength
            *
            0.28
        )

    # --------------------------------------------------------
    # JUMP
    # --------------------------------------------------------

    elif motion == "Jump":

        jump = math.sin(
            math.pi
            *
            t
        )

        ty = (
            -jump
            *
            strength
            *
            1.6
        )

        if depth > 0:

            angle = (
                sine
                *
                strength
                *
                0.25
            )

    # --------------------------------------------------------
    # DANCE
    # --------------------------------------------------------

    elif motion == "Dance":

        angle = (
            math.sin(
                4
                *
                math.pi
                *
                t
                +
                depth
                *
                0.5
            )
            *
            strength
            *
            1.15
        )

        tx = (
            math.sin(
                2
                *
                math.pi
                *
                t
                +
                depth
            )
            *
            strength
            *
            0.25
        )

        ty = (
            math.cos(
                4
                *
                math.pi
                *
                t
            )
            *
            strength
            *
            0.18
        )

    # --------------------------------------------------------
    # BOUNCE
    # --------------------------------------------------------

    elif motion == "Bounce":

        bounce = abs(
            math.sin(
                math.pi
                *
                t
            )
        )

        ty = (
            -bounce
            *
            strength
            *
            1.15
        )

        angle = (
            sine
            *
            strength
            *
            0.30
        )

    # --------------------------------------------------------
    # SHAKE
    # --------------------------------------------------------

    elif motion == "Shake":

        angle = (
            math.sin(
                10
                *
                math.pi
                *
                t
            )
            *
            strength
            *
            0.50
        )

        tx = (
            math.sin(
                14
                *
                math.pi
                *
                t
            )
            *
            strength
            *
            0.25
        )

    # --------------------------------------------------------
    # FLOAT
    # --------------------------------------------------------

    elif motion == "Float":

        ty = (
            sine
            *
            strength
            *
            0.60
        )

        tx = (
            cosine
            *
            strength
            *
            0.25
        )

        angle = (
            sine
            *
            strength
            *
            0.25
        )

    # --------------------------------------------------------
    # CELEBRATE
    # --------------------------------------------------------

    elif motion == "Celebrate":

        angle = (
            sine
            *
            strength
            *
            1.25
        )

        ty = (
            -abs(sine)
            *
            strength
            *
            0.42
        )

        if depth > 0:

            tx = (
                math.sin(
                    4
                    *
                    math.pi
                    *
                    t
                )
                *
                strength
                *
                0.35
            )

    # --------------------------------------------------------
    # MOVE ACROSS CANVAS
    # --------------------------------------------------------

    elif motion == "Move Across Canvas":

        travel = (
            -canvas_width * 0.35
            +
            t
            *
            canvas_width
            *
            0.70
        )

        tx = (
            travel
            *
            (
                1.0
                if depth == 0
                else 0.02
            )
        )

        # tiny natural bobbing
        ty = (
            math.sin(
                phase * 2
            )
            *
            strength
            *
            0.12
        )

    # --------------------------------------------------------
    # EXIT RIGHT
    # --------------------------------------------------------

    elif motion == "Exit Right":

        eased = smoothstep(t)

        tx = (
            eased
            *
            canvas_width
            *
            1.20
        )

        ty = (
            math.sin(
                phase * 2
            )
            *
            strength
            *
            0.10
        )

    # --------------------------------------------------------
    # EXIT LEFT
    # --------------------------------------------------------

    elif motion == "Exit Left":

        eased = smoothstep(t)

        tx = (
            -eased
            *
            canvas_width
            *
            1.20
        )

        ty = (
            math.sin(
                phase * 2
            )
            *
            strength
            *
            0.10
        )

    return (
        angle,
        tx,
        ty,
    )


# ============================================================
# WORLD TRANSFORMS
# ============================================================

def compute_world_transforms(
    parts,
    motion,
    t,
    intensity,
    canvas_width,
    canvas_height,
):
    """
    Parent transforms are inherited by children.

    Example:

        body
          |
          arm
           |
          hand

    Moving the body therefore moves the arm and hand too.
    """

    transforms = {}

    for part in parts:

        angle, tx, ty = motion_for_part(
            motion,
            part,
            t,
            intensity,
            canvas_width,
            canvas_height,
        )

        pivot = part["base"]

        local_M = get_affine_matrix(
            angle,
            tx,
            ty,
            pivot,
        )

        parent_id = part["parent"]

        if (
            parent_id is not None
            and parent_id in transforms
        ):

            world_M = (
                transforms[parent_id]
                @
                local_M
            )

        else:

            world_M = local_M

        transforms[
            part["id"]
        ] = world_M

    return transforms


# ============================================================
# PART LAYER PREPARATION
# ============================================================

def prepare_part_layers(
    original,
    parts,
):
    """
    Store each part's original pixels and alpha mask.

    This is the important frame-by-frame change.

    We don't repeatedly transform the whole original image.
    We transform only the pixels belonging to the part.
    """

    layers = {}

    for part in parts:

        mask = (
            part["mask"]
            .astype(np.uint8)
            * 255
        )

        # Slight feathering prevents jagged edges.
        alpha = cv2.GaussianBlur(
            mask,
            (3, 3),
            0,
        )

        layer = original.copy()

        layers[
            part["id"]
        ] = {
            "image": layer,
            "mask": alpha,
        }

    return layers


# ============================================================
# ALPHA COMPOSITING
# ============================================================

def alpha_composite(
    canvas,
    layer,
    alpha,
):
    """
    Standard alpha compositing.
    """

    alpha_f = (
        alpha.astype(
            np.float32
        )
        /
        255.0
    )

    alpha_f = alpha_f[:, :, None]

    result = (
        layer.astype(
            np.float32
        )
        *
        alpha_f
        +
        canvas.astype(
            np.float32
        )
        *
        (
            1.0
            -
            alpha_f
        )
    )

    return np.clip(
        result,
        0,
        255,
    ).astype(
        np.uint8
    )


# ============================================================
# RENDER ONE FRAME
# ============================================================

def render_frame(
    original,
    background,
    parts,
    part_layers,
    motion,
    t,
    intensity,
):
    """
    Render exactly ONE animation frame.

    Every frame is a fresh image.

    This is effectively:

        original drawing
             ↓
        extracted parts
             ↓
        tiny movement
             ↓
        frame N

    Then frame N+1 gets a slightly different movement.
    """

    canvas = background.copy()

    h, w = original.shape[:2]

    transforms = compute_world_transforms(
        parts,
        motion,
        t,
        intensity,
        w,
        h,
    )

    # Parents first, children afterward.
    ordered_parts = sorted(
        parts,
        key=lambda p: p["depth"],
    )

    for part in ordered_parts:

        part_id = part["id"]

        layer_data = part_layers[
            part_id
        ]

        source_image = (
            layer_data["image"]
        )

        source_mask = (
            layer_data["mask"]
        )

        affine = (
            transforms[part_id]
            [:2, :]
        )

        warped_image = cv2.warpAffine(
            source_image,
            affine,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(
                0,
                0,
                0,
            ),
        )

        warped_mask = cv2.warpAffine(
            source_mask,
            affine,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

        # Prevent extremely faint interpolation ghosts.
        warped_mask = np.where(
            warped_mask < 5,
            0,
            warped_mask,
        ).astype(
            np.uint8
        )

        canvas = alpha_composite(
            canvas,
            warped_image,
            warped_mask,
        )

    return canvas


# ============================================================
# RIG VISUALIZATION
# ============================================================

def draw_rig(
    image,
    parts,
):
    preview = image.copy()

    colors = [
        (0, 0, 255),
        (0, 180, 0),
        (255, 0, 0),
        (0, 180, 180),
        (180, 0, 180),
        (255, 120, 0),
    ]

    for index, part in enumerate(parts):

        color = colors[
            index
            %
            len(colors)
        ]

        base = (
            int(part["base"][0]),
            int(part["base"][1]),
        )

        tip = (
            int(part["tip"][0]),
            int(part["tip"][1]),
        )

        center = (
            int(part["center"][0]),
            int(part["center"][1]),
        )

        cv2.line(
            preview,
            base,
            tip,
            color,
            3,
        )

        cv2.circle(
            preview,
            base,
            8,
            (0, 0, 255),
            -1,
        )

        cv2.circle(
            preview,
            tip,
            6,
            (0, 255, 0),
            -1,
        )

        cv2.circle(
            preview,
            center,
            4,
            color,
            -1,
        )

        cv2.putText(
            preview,
            str(index + 1),
            (
                center[0] + 8,
                center[1] - 8,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return preview


# ============================================================
# FRAME DIFFERENCE PREVIEW
# ============================================================

def make_motion_sheet(
    frames,
):
    """
    Create a small contact sheet showing several frames.

    This is useful for checking whether the movement is actually
    changing gradually rather than jumping.
    """

    if not frames:
        return None

    count = min(
        6,
        len(frames),
    )

    indices = np.linspace(
        0,
        len(frames) - 1,
        count,
    ).astype(int)

    selected = [
        frames[i]
        for i in indices
    ]

    thumbnails = []

    for frame in selected:

        thumb = cv2.resize(
            frame,
            (260, 260),
            interpolation=cv2.INTER_AREA,
        )

        thumbnails.append(
            thumb
        )

    rows = []

    for i in range(
        0,
        len(thumbnails),
        3,
    ):

        row = thumbnails[
            i:i + 3
        ]

        while len(row) < 3:

            row.append(
                np.ones_like(
                    thumbnails[0]
                )
                *
                255
            )

        rows.append(
            np.hstack(row)
        )

    return np.vstack(rows)


# ============================================================
# GIF EXPORT
# ============================================================

def build_gif(
    frames,
    duration,
):
    if not frames:
        return None

    buffer = io.BytesIO()

    prepared = []

    for frame in frames:

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB,
        )

        prepared.append(
            Image.fromarray(
                rgb
            ).convert(
                "P",
                palette=Image.ADAPTIVE,
            )
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
# MP4 EXPORT
# ============================================================

def build_mp4(
    frames,
    fps,
):
    if not frames:
        return None

    h, w = frames[0].shape[:2]

    with tempfile.NamedTemporaryFile(
        suffix=".mp4",
        delete=False,
    ) as tmp:

        temp_path = tmp.name

    try:

        writer = cv2.VideoWriter(
            temp_path,
            cv2.VideoWriter_fourcc(
                *"mp4v"
            ),
            fps,
            (w, h),
        )

        if not writer.isOpened():
            return None

        for frame in frames:
            writer.write(frame)

        writer.release()

        with open(
            temp_path,
            "rb",
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
# MAIN APPLICATION
# ============================================================

uploaded = st.file_uploader(
    "Upload your hand-drawn character sheet",
    type=[
        "jpg",
        "jpeg",
        "png",
        "webp",
    ],
)


if uploaded is not None:

    try:

        # ====================================================
        # READ IMAGE
        # ====================================================

        file_bytes = np.asarray(
            bytearray(
                uploaded.read()
            ),
            dtype=np.uint8,
        )

        image = cv2.imdecode(
            file_bytes,
            cv2.IMREAD_COLOR,
        )

        if image is None:

            st.error(
                "Could not read the uploaded image."
            )

            st.stop()

        original_h, original_w = (
            image.shape[:2]
        )

        if max(
            original_h,
            original_w,
        ) > MAX_IMAGE_SIZE:

            scale = (
                MAX_IMAGE_SIZE
                /
                max(
                    original_h,
                    original_w,
                )
            )

            image = cv2.resize(
                image,
                (
                    int(
                        original_w
                        *
                        scale
                    ),
                    int(
                        original_h
                        *
                        scale
                    ),
                ),
                interpolation=cv2.INTER_AREA,
            )

        # ====================================================
        # SIDEBAR
        # ====================================================

        st.sidebar.header(
            "🎨 Part Detection"
        )

        detected_colors = (
            extract_dominant_colors(
                image
            )
        )

        if not detected_colors:

            st.warning(
                "No clear colored regions were detected."
            )

            st.image(
                cv2.cvtColor(
                    image,
                    cv2.COLOR_BGR2RGB,
                ),
                use_container_width=True,
            )

            st.stop()

        color_dict = {
            c["label"]: c
            for c in detected_colors
        }

        default_count = min(
            3,
            len(color_dict),
        )

        selected_labels = (
            st.sidebar.multiselect(
                "Select body/part colors",
                list(
                    color_dict.keys()
                ),
                default=list(
                    color_dict.keys()
                )[:default_count],
            )
        )

        selected_colors = [
            color_dict[label]
            for label in selected_labels
        ]

        st.sidebar.header(
            "🎬 Animation"
        )

        motion = st.sidebar.selectbox(
            "Animation Style",
            MOTIONS,
        )

        intensity = st.sidebar.slider(
            "Motion Strength",
            1,
            40,
            12,
        )

        fps = st.sidebar.select_slider(
            "Playback FPS",
            options=[
                8,
                10,
                12,
                15,
                20,
                24,
                30,
            ],
            value=12,
        )

        duration_seconds = (
            st.sidebar.slider(
                "Animation Duration",
                1.0,
                8.0,
                2.5,
                0.5,
            )
        )

        st.sidebar.header(
            "🎞️ Frame Generation"
        )

        frame_multiplier = (
            st.sidebar.select_slider(
                "Frame Density",
                options=[
                    1,
                    2,
                    3,
                ],
                value=1,
                help=(
                    "1 = normal frame count. "
                    "2 = twice as many generated frames. "
                    "3 = very smooth but slower."
                ),
            )
        )

        # Actual frames shown in the final video.
        frame_count = max(
            8,
            int(
                fps
                *
                duration_seconds
            ),
        )

        # Extra generated frames can be useful for
        # smoother GIF motion.
        render_frame_count = (
            frame_count
            *
            frame_multiplier
        )

        parts = detect_parts(
            image,
            selected_colors,
        )

        # ====================================================
        # PREVIEWS
        # ====================================================

        col1, col2 = st.columns(2)

        with col1:

            st.subheader(
                "🖼️ Original Drawing"
            )

            st.image(
                cv2.cvtColor(
                    image,
                    cv2.COLOR_BGR2RGB,
                ),
                use_container_width=True,
            )

        with col2:

            st.subheader(
                f"🦴 Detected Rig ({len(parts)} Parts)"
            )

            if parts:

                rig = draw_rig(
                    image,
                    parts,
                )

                st.image(
                    cv2.cvtColor(
                        rig,
                        cv2.COLOR_BGR2RGB,
                    ),
                    use_container_width=True,
                )

            else:

                st.warning(
                    "No sufficiently large parts detected."
                )

        # ====================================================
        # INFORMATION
        # ====================================================

        if parts:

            st.info(
                f"""
**Frame-by-frame engine ready**

Detected: **{len(parts)} parts**

Output playback: **{fps} FPS**

Duration: **{duration_seconds:.1f} seconds**

Generated frames: **{render_frame_count}**

The program will create a new slightly different pose
for every frame instead of simply moving one image.
"""
            )

        # ====================================================
        # GENERATE
        # ====================================================

        generate = st.button(
            "✨ Generate Frame-by-Frame Animation",
            type="primary",
            use_container_width=True,
        )

        if (
            generate
            and parts
        ):

            progress = st.progress(
                0,
                text="Preparing artwork...",
            )

            # =================================================
            # BACKGROUND CACHE
            # =================================================

            bg_key = (
                "background_"
                + str(
                    hash(
                        uploaded.name
                        +
                        str(
                            len(parts)
                        )
                        +
                        str(
                            image.shape
                        )
                    )
                )
            )

            if (
                bg_key
                not in st.session_state
            ):

                st.session_state[
                    bg_key
                ] = prepare_background(
                    image,
                    parts,
                )

            background = (
                st.session_state[
                    bg_key
                ]
            )

            progress.progress(
                15,
                text="Preparing original drawing layers...",
            )

            # =================================================
            # PREPARE ORIGINAL PARTS
            # =================================================

            part_layers = (
                prepare_part_layers(
                    image,
                    parts,
                )
            )

            # =================================================
            # RENDER FRAMES
            # =================================================

            frames = []

            for i in range(
                render_frame_count
            ):

                if render_frame_count <= 1:

                    t = 0.0

                else:

                    t = (
                        i
                        /
                        (
                            render_frame_count
                            -
                            1
                        )
                    )

                frame = render_frame(
                    image,
                    background,
                    parts,
                    part_layers,
                    motion,
                    t,
                    intensity,
                )

                frames.append(
                    frame
                )

                percent = (
                    20
                    +
                    int(
                        70
                        *
                        (
                            i + 1
                        )
                        /
                        render_frame_count
                    )
                )

                progress.progress(
                    percent,
                    text=(
                        f"Drawing Frame "
                        f"{i + 1}/"
                        f"{render_frame_count}"
                    ),
                )

            progress.progress(
                92,
                text="Building animation...",
            )

            # =================================================
            # EXPORT
            # =================================================

            # If frame density > 1, keep the actual playback
            # duration approximately correct by using the
            # corresponding GIF duration.
            gif_duration = max(
                1,
                int(
                    1000
                    /
                    (
                        fps
                        *
                        frame_multiplier
                    )
                ),
            )

            gif_data = build_gif(
                frames,
                gif_duration,
            )

            progress.progress(
                100,
                text="Animation complete!",
            )

            progress.empty()

            # =================================================
            # MOTION PREVIEW
            # =================================================

            st.subheader(
                "🎬 Generated Animation"
            )

            if gif_data:

                st.image(
                    gif_data,
                    use_container_width=True,
                )

            # =================================================
            # FRAME CONTACT SHEET
            # =================================================

            with st.expander(
                "🔍 Inspect Individual Generated Frames"
            ):

                motion_sheet = (
                    make_motion_sheet(
                        frames
                    )
                )

                if motion_sheet is not None:

                    st.image(
                        cv2.cvtColor(
                            motion_sheet,
                            cv2.COLOR_BGR2RGB,
                        ),
                        caption=(
                            "The animation is made from "
                            "many small changes between frames."
                        ),
                        use_container_width=True,
                    )

            # =================================================
            # DOWNLOADS
            # =================================================

            safe_name = (
                motion
                .lower()
                .replace(
                    " ",
                    "_",
                )
            )

            d1, d2 = st.columns(2)

            with d1:

                if gif_data:

                    st.download_button(
                        "⬇️ Download GIF",
                        data=gif_data,
                        file_name=(
                            f"{safe_name}.gif"
                        ),
                        mime="image/gif",
                        use_container_width=True,
                    )

            with d2:

                mp4_data = build_mp4(
                    frames,
                    fps
                    *
                    frame_multiplier,
                )

                if mp4_data:

                    st.download_button(
                        "🎞️ Download MP4",
                        data=mp4_data,
                        file_name=(
                            f"{safe_name}.mp4"
                        ),
                        mime="video/mp4",
                        use_container_width=True,
                    )

            # =================================================
            # TECHNICAL INFORMATION
            # =================================================

            st.success(
                f"""
Animation generated successfully.

**Parts:** {len(parts)}

**Frames:** {len(frames)}

**Playback FPS:** {fps}

**Duration:** approximately {duration_seconds:.1f} seconds

**Technique:** original-pixel part extraction + incremental
frame-by-frame transformations + alpha compositing.
"""
            )

    except Exception as e:

        st.error(
            f"❌ Processing Error: {e}"
        )

        st.exception(e)

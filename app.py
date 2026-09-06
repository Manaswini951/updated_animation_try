import io
import math
import base64
import tempfile
import os

import cv2
import numpy as np
import streamlit as st
from PIL import Image


# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="Hand-Drawn Puppet Animator",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🎨 Hand-Drawn Puppet Animator")

st.markdown(
    """
Upload a hand-drawn character and use colored regions as automatic
animation parts. The app detects attachment points and creates
smooth puppet-style motion while trying to keep the original artwork intact.
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

    try:

        small = cv2.resize(
            image,
            (160, 160),
            interpolation=cv2.INTER_AREA
        )

        hsv_small = cv2.cvtColor(
            small,
            cv2.COLOR_BGR2HSV
        )

        saturation = hsv_small[:, :, 1]

        # Only look for reasonably saturated colors.
        valid = saturation > 45

        pixels = small[valid].reshape(-1, 3)

        if len(pixels) < 50:
            return []

        pixels = pixels.astype(np.float32)

        k = min(
            num_clusters,
            max(2, len(pixels) // 80)
        )

        criteria = (
            cv2.TERM_CRITERIA_EPS +
            cv2.TERM_CRITERIA_MAX_ITER,
            30,
            1.0
        )

        _, labels, centers = cv2.kmeans(
            pixels,
            k,
            None,
            criteria,
            5,
            cv2.KMEANS_PP_CENTERS
        )

        counts = np.bincount(
            labels.flatten(),
            minlength=k
        )

        total = max(
            1,
            int(np.sum(counts))
        )

        detected = []

        for idx, center in enumerate(centers):

            b, g, r = [int(x) for x in center]

            hsv = cv2.cvtColor(
                np.uint8([[[b, g, r]]]),
                cv2.COLOR_BGR2HSV
            )[0][0]

            h_val = int(hsv[0])
            s_val = int(hsv[1])
            v_val = int(hsv[2])

            coverage = (
                counts[idx] /
                total *
                100
            )

            if coverage < 2.0:
                continue

            if s_val < 45:
                continue

            hex_code = (
                f"#{r:02x}{g:02x}{b:02x}"
            )

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
                        v_val
                    ),
                    "hex": hex_code,
                    "rgb": (r, g, b),
                    "coverage": coverage,
                }
            )

        detected.sort(
            key=lambda x: x["coverage"],
            reverse=True
        )

        return detected

    except Exception:
        return []


# ============================================================
# COLOR MASK
# ============================================================

def hue_distance(a, b):
    d = abs(a - b)
    return min(d, 180 - d)


def make_color_mask(image, color):

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV
    )

    h0, s0, v0 = color["hsv"]

    # More tolerant than the original script.
    hue_width = 18

    if h0 - hue_width < 0:

        lower1 = np.array(
            [0, max(35, s0 - 90), max(30, v0 - 90)],
            dtype=np.uint8
        )

        upper1 = np.array(
            [h0 + hue_width, 255, 255],
            dtype=np.uint8
        )

        lower2 = np.array(
            [180 + h0 - hue_width, max(35, s0 - 90), max(30, v0 - 90)],
            dtype=np.uint8
        )

        upper2 = np.array(
            [179, 255, 255],
            dtype=np.uint8
        )

        mask = cv2.bitwise_or(
            cv2.inRange(hsv, lower1, upper1),
            cv2.inRange(hsv, lower2, upper2)
        )

    elif h0 + hue_width > 179:

        lower1 = np.array(
            [h0 - hue_width, max(35, s0 - 90), max(30, v0 - 90)],
            dtype=np.uint8
        )

        upper1 = np.array(
            [179, 255, 255],
            dtype=np.uint8
        )

        lower2 = np.array(
            [0, max(35, s0 - 90), max(30, v0 - 90)],
            dtype=np.uint8
        )

        upper2 = np.array(
            [(h0 + hue_width) - 180, 255, 255],
            dtype=np.uint8
        )

        mask = cv2.bitwise_or(
            cv2.inRange(hsv, lower1, upper1),
            cv2.inRange(hsv, lower2, upper2)
        )

    else:

        lower = np.array(
            [
                max(0, h0 - hue_width),
                max(35, s0 - 90),
                max(25, v0 - 90),
            ],
            dtype=np.uint8
        )

        upper = np.array(
            [
                min(179, h0 + hue_width),
                255,
                255,
            ],
            dtype=np.uint8
        )

        mask = cv2.inRange(
            hsv,
            lower,
            upper
        )

    # Clean camera noise.
    kernel_small = np.ones(
        (3, 3),
        np.uint8
    )

    kernel_medium = np.ones(
        (5, 5),
        np.uint8
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel_small
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel_medium
    )

    return mask


# ============================================================
# PART DETECTION
# ============================================================

def component_boundary(mask):

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE
    )

    if not contours:
        return np.empty((0, 2), dtype=np.float32)

    points = np.vstack(
        [
            c.reshape(-1, 2)
            for c in contours
        ]
    )

    return points.astype(np.float32)


def farthest_point(points, origin):

    if len(points) == 0:
        return origin

    diff = points - np.array(
        origin,
        dtype=np.float32
    )

    dist = np.sum(
        diff * diff,
        axis=1
    )

    return tuple(
        points[np.argmax(dist)]
    )


def closest_points_between_masks(mask_a, mask_b):

    pts_a = component_boundary(mask_a)
    pts_b = component_boundary(mask_b)

    if len(pts_a) == 0 or len(pts_b) == 0:
        return None, None, float("inf")

    # Reduce computational cost for large shapes.
    max_points = 500

    if len(pts_a) > max_points:
        idx = np.linspace(
            0,
            len(pts_a) - 1,
            max_points
        ).astype(int)

        pts_a = pts_a[idx]

    if len(pts_b) > max_points:
        idx = np.linspace(
            0,
            len(pts_b) - 1,
            max_points
        ).astype(int)

        pts_b = pts_b[idx]

    diff = (
        pts_a[:, None, :] -
        pts_b[None, :, :]
    )

    dist2 = np.sum(
        diff * diff,
        axis=2
    )

    ia, ib = np.unravel_index(
        np.argmin(dist2),
        dist2.shape
    )

    p1 = tuple(
        pts_a[ia]
    )

    p2 = tuple(
        pts_b[ib]
    )

    return (
        p1,
        p2,
        float(math.sqrt(dist2[ia, ib]))
    )


def detect_parts(image, selected_colors):

    h, w = image.shape[:2]

    all_parts = []

    for color_index, color in enumerate(
        selected_colors
    ):

        mask = make_color_mask(
            image,
            color
        )

        num_labels, labels, stats, centroids = (
            cv2.connectedComponentsWithStats(
                mask,
                connectivity=8
            )
        )

        min_area = max(
            250,
            int(h * w * 0.00025)
        )

        for i in range(
            1,
            num_labels
        ):

            area = int(
                stats[i, cv2.CC_STAT_AREA]
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

            # Principal-axis endpoints.
            pts = boundary.astype(
                np.float32
            )

            mean = np.mean(
                pts,
                axis=0
            )

            centered = pts - mean

            covariance = np.cov(
                centered.T
            )

            eigenvalues, eigenvectors = (
                np.linalg.eigh(covariance)
            )

            axis = eigenvectors[
                :, np.argmax(eigenvalues)
            ]

            projections = (
                centered @ axis
            )

            p_min = tuple(
                pts[np.argmin(projections)]
            )

            p_max = tuple(
                pts[np.argmax(projections)]
            )

            length = float(
                np.linalg.norm(
                    np.array(p_max) -
                    np.array(p_min)
                )
            )

            if length < 10:
                length = 10

            all_parts.append(
                {
                    "id": len(all_parts),
                    "color_index": color_index,
                    "color_name": color["label"],
                    "mask": component_mask > 0,
                    "area": area,
                    "center": (
                        float(cx),
                        float(cy)
                    ),
                    "bbox": (
                        x,
                        y,
                        cw,
                        ch
                    ),
                    "axis_a": p_min,
                    "axis_b": p_max,
                    "length": length,
                    "base": (
                        float(cx),
                        float(cy)
                    ),
                    "tip": p_max,
                    "parent": None,
                    "depth": 0,
                }
            )

    if not all_parts:
        return []

    # --------------------------------------------------------
    # Determine parent-child relationships.
    #
    # A smaller colored part usually attaches to a larger
    # nearby colored part.
    # --------------------------------------------------------

    for part in all_parts:

        best_parent = None
        best_distance = float("inf")

        for candidate in all_parts:

            if candidate["id"] == part["id"]:
                continue

            # Parent should normally be larger.
            if candidate["area"] < part["area"] * 0.30:
                continue

            p1, p2, distance = (
                closest_points_between_masks(
                    part["mask"],
                    candidate["mask"]
                )
            )

            if p1 is None:
                continue

            # Attachment distance tolerance.
            tolerance = max(
                80,
                min(
                    w,
                    h
                ) * 0.12
            )

            if distance < tolerance:

                if distance < best_distance:
                    best_distance = distance
                    best_parent = candidate
                    best_p1 = p1
                    best_p2 = p2

        if best_parent is not None:

            part["parent"] = best_parent["id"]

            # Point on child closest to parent.
            part["base"] = (
                float(best_p1[0]),
                float(best_p1[1])
            )

            # Farthest point on child from base.
            child_boundary = (
                component_boundary(
                    (
                        part["mask"].astype(
                            np.uint8
                        ) * 255
                    )
                )
            )

            part["tip"] = farthest_point(
                child_boundary,
                part["base"]
            )

            part["attachment_distance"] = (
                best_distance
            )

        else:

            # Root.
            part["parent"] = None

            # Root doesn't rotate around a random edge.
            part["base"] = part["center"]

            part["tip"] = farthest_point(
                component_boundary(
                    (
                        part["mask"].astype(
                            np.uint8
                        ) * 255
                    )
                ),
                part["center"]
            )

    # --------------------------------------------------------
    # Calculate hierarchy depth.
    # --------------------------------------------------------

    def get_depth(part):

        visited = set()

        current = part

        depth = 0

        while (
            current["parent"] is not None
            and depth < 20
        ):

            if current["id"] in visited:
                break

            visited.add(
                current["id"]
            )

            parent_id = current["parent"]

            parent = next(
                (
                    p for p in all_parts
                    if p["id"] == parent_id
                ),
                None
            )

            if parent is None:
                break

            current = parent
            depth += 1

        return depth

    for part in all_parts:
        part["depth"] = get_depth(part)

    # Largest first.
    all_parts.sort(
        key=lambda p: p["area"],
        reverse=True
    )

    return all_parts


# ============================================================
# IMAGE COMPOSITING
# ============================================================

def rotate_image(image, angle):

    if angle == 0:
        return image

    if angle == 90:
        return cv2.rotate(
            image,
            cv2.ROTATE_90_CLOCKWISE
        )

    if angle == 180:
        return cv2.rotate(
            image,
            cv2.ROTATE_180
        )

    if angle == 270:
        return cv2.rotate(
            image,
            cv2.ROTATE_90_COUNTERCLOCKWISE
        )

    return image


def prepare_background(original, parts):

    if not parts:
        return original.copy()

    combined = np.zeros(
        original.shape[:2],
        dtype=np.uint8
    )

    for part in parts:

        mask = (
            part["mask"].astype(
                np.uint8
            ) * 255
        )

        # Only a modest expansion.
        mask = cv2.dilate(
            mask,
            np.ones((3, 3), np.uint8),
            iterations=1
        )

        combined = cv2.bitwise_or(
            combined,
            mask
        )

    # Don't destroy huge regions.
    area_ratio = (
        np.count_nonzero(combined) /
        float(combined.size)
    )

    if area_ratio > 0.55:
        return original.copy()

    try:

        background = cv2.inpaint(
            original,
            combined,
            5,
            cv2.INPAINT_TELEA
        )

        return background

    except Exception:

        return original.copy()


def rotate_point(point, center, angle_rad):

    x, y = point
    cx, cy = center

    c = math.cos(angle_rad)
    s = math.sin(angle_rad)

    nx = (
        cx +
        (x - cx) * c -
        (y - cy) * s
    )

    ny = (
        cy +
        (x - cx) * s +
        (y - cy) * c
    )

    return nx, ny


def transform_part(
    original,
    part,
    angle_deg,
    tx,
    ty
):

    mask = (
        part["mask"].astype(
            np.uint8
        ) * 255
    )

    # Use a slightly expanded mask so edges
    # don't become visibly cut.
    mask = cv2.dilate(
        mask,
        np.ones((3, 3), np.uint8),
        iterations=1
    )

    ys, xs = np.where(
        mask > 0
    )

    if len(xs) == 0:
        return None, None

    x1 = max(
        0,
        int(xs.min()) - 10
    )

    y1 = max(
        0,
        int(ys.min()) - 10
    )

    x2 = min(
        original.shape[1],
        int(xs.max()) + 11
    )

    y2 = min(
        original.shape[0],
        int(ys.max()) + 11
    )

    crop = original[
        y1:y2,
        x1:x2
    ]

    crop_mask = mask[
        y1:y2,
        x1:x2
    ]

    base = part["base"]

    local_base = (
        base[0] - x1,
        base[1] - y1
    )

    center = local_base

    matrix = cv2.getRotationMatrix2D(
        center,
        angle_deg,
        1.0
    )

    matrix[0, 2] += tx
    matrix[1, 2] += ty

    warped = cv2.warpAffine(
        crop,
        matrix,
        (
            crop.shape[1],
            crop.shape[0]
        ),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255)
    )

    warped_mask = cv2.warpAffine(
        crop_mask,
        matrix,
        (
            crop.shape[1],
            crop.shape[0]
        ),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )

    return (
        warped,
        warped_mask,
        x1,
        y1
    )


def paste_layer(
    canvas,
    layer,
    layer_mask,
    x,
    y
):

    h, w = canvas.shape[:2]

    lh, lw = layer.shape[:2]

    x1 = max(
        0,
        x
    )

    y1 = max(
        0,
        y
    )

    x2 = min(
        w,
        x + lw
    )

    y2 = min(
        h,
        y + lh
    )

    if x1 >= x2 or y1 >= y2:
        return canvas

    lx1 = x1 - x
    ly1 = y1 - y

    lx2 = lx1 + (x2 - x1)
    ly2 = ly1 + (y2 - y1)

    roi = canvas[
        y1:y2,
        x1:x2
    ]

    src = layer[
        ly1:ly2,
        lx1:lx2
    ]

    alpha = (
        layer_mask[
            ly1:ly2,
            lx1:lx2
        ].astype(
            np.float32
        ) / 255.0
    )

    alpha = alpha[:, :, None]

    result = (
        src.astype(np.float32) * alpha +
        roi.astype(np.float32) * (1 - alpha)
    )

    canvas[
        y1:y2,
        x1:x2
    ] = np.clip(
        result,
        0,
        255
    ).astype(np.uint8)

    return canvas


# ============================================================
# MOTION ENGINE
# ============================================================

def smoothstep(t):

    return (
        t * t *
        (3.0 - 2.0 * t)
    )


def motion_for_part(
    motion,
    part,
    normalized_time,
    intensity
):

    t = normalized_time

    # Stable oscillators.
    sine = math.sin(
        2 * math.pi * t
    )

    cosine = math.cos(
        2 * math.pi * t
    )

    # Parts alternate naturally.
    side = (
        1
        if part["id"] % 2 == 0
        else -1
    )

    depth = part.get(
        "depth",
        0
    )

    # Children move slightly more.
    depth_factor = (
        0.65 +
        min(depth, 3) * 0.15
    )

    strength = (
        intensity *
        depth_factor
    )

    angle = 0.0
    tx = 0.0
    ty = 0.0

    # --------------------------------------------------------
    # IDLE
    # --------------------------------------------------------

    if motion == "Idle Breathing":

        if depth == 0:
            ty = (
                sine *
                strength *
                0.12
            )

            angle = (
                sine *
                strength *
                0.10
            )

        else:
            angle = (
                sine *
                strength *
                0.30
            )

    # --------------------------------------------------------
    # WAVE
    # --------------------------------------------------------

    elif motion == "Wave":

        angle = (
            sine *
            strength *
            0.75
        )

        tx = (
            sine *
            strength *
            0.10
        )

        if depth >= 2:
            angle *= 1.4

    # --------------------------------------------------------
    # WALK
    # --------------------------------------------------------

    elif motion == "Walk":

        if depth == 0:

            ty = (
                abs(
                    math.sin(
                        2 * math.pi * t
                    )
                ) *
                strength *
                0.18
            )

            angle = (
                sine *
                strength *
                0.12
            )

        else:

            phase = (
                2 * math.pi * t +
                (math.pi if side < 0 else 0)
            )

            angle = (
                math.sin(phase) *
                strength *
                1.0
            )

            ty = (
                max(
                    0,
                    -math.cos(phase)
                ) *
                strength *
                0.18
            )

    # --------------------------------------------------------
    # RUN
    # --------------------------------------------------------

    elif motion == "Run":

        if depth == 0:

            ty = (
                abs(
                    math.sin(
                        4 * math.pi * t
                    )
                ) *
                strength *
                0.35
            )

            angle = (
                sine *
                strength *
                0.25
            )

        else:

            phase = (
                4 * math.pi * t +
                (math.pi if side < 0 else 0)
            )

            angle = (
                math.sin(phase) *
                strength *
                1.5
            )

            ty = (
                max(
                    0,
                    -math.cos(phase)
                ) *
                strength *
                0.35
            )

    # --------------------------------------------------------
    # JUMP
    # --------------------------------------------------------

    elif motion == "Jump":

        # One smooth jump arc.
        jump = math.sin(
            math.pi * t
        )

        ty = (
            -jump *
            strength *
            1.5
        )

        if depth > 0:
            angle = (
                sine *
                strength *
                0.30
            )

    # --------------------------------------------------------
    # DANCE
    # --------------------------------------------------------

    elif motion == "Dance":

        angle = (
            math.sin(
                4 * math.pi * t +
                depth * 0.5
            ) *
            strength *
            1.0
        )

        tx = (
            math.sin(
                2 * math.pi * t +
                depth
            ) *
            strength *
            0.25
        )

        ty = (
            math.cos(
                4 * math.pi * t
            ) *
            strength *
            0.20
        )

    # --------------------------------------------------------
    # BOUNCE
    # --------------------------------------------------------

    elif motion == "Bounce":

        bounce = abs(
            math.sin(
                math.pi * t
            )
        )

        ty = (
            -bounce *
            strength *
            1.0
        )

        angle = (
            sine *
            strength *
            0.35
        )

    # --------------------------------------------------------
    # SHAKE
    # --------------------------------------------------------

    elif motion == "Shake":

        angle = (
            math.sin(
                10 * math.pi * t
            ) *
            strength *
            0.55
        )

        tx = (
            math.sin(
                14 * math.pi * t
            ) *
            strength *
            0.25
        )

    # --------------------------------------------------------
    # FLOAT
    # --------------------------------------------------------

    elif motion == "Float":

        ty = (
            math.sin(
                2 * math.pi * t
            ) *
            strength *
            0.65
        )

        tx = (
            math.cos(
                2 * math.pi * t
            ) *
            strength *
            0.25
        )

        angle = (
            sine *
            strength *
            0.30
        )

    # --------------------------------------------------------
    # CELEBRATE
    # --------------------------------------------------------

    elif motion == "Celebrate":

        angle = (
            sine *
            strength *
            1.2
        )

        ty = (
            -abs(sine) *
            strength *
            0.45
        )

        if depth > 0:
            tx = (
                math.sin(
                    4 * math.pi * t
                ) *
                strength *
                0.35
            )

    return (
        angle,
        tx,
        ty
    )


# ============================================================
# FRAME GENERATION
# ============================================================

def render_frame(
    original,
    background,
    parts,
    motion,
    t,
    intensity
):

    canvas = background.copy()

    # Process roots first.
    ordered_parts = sorted(
        parts,
        key=lambda p: p["depth"]
    )

    for part in ordered_parts:

        angle, tx, ty = (
            motion_for_part(
                motion,
                part,
                t,
                intensity
            )
        )

        result = transform_part(
            original,
            part,
            angle,
            tx,
            ty
        )

        if result is None:
            continue

        layer, layer_mask, x, y = result

        canvas = paste_layer(
            canvas,
            layer,
            layer_mask,
            x,
            y
        )

    return canvas


# ============================================================
# GIF
# ============================================================

def build_gif(
    frames,
    duration
):

    buffer = io.BytesIO()

    prepared = []

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
        optimize=False
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

    buffer = io.BytesIO()

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

        for frame in frames:
            writer.write(frame)

        writer.release()

        with open(
            temp_path,
            "rb"
        ) as f:
            data = f.read()

        return data

    except Exception:
        return None

    finally:

        if os.path.exists(
            temp_path
        ):
            os.remove(
                temp_path
            )


# ============================================================
# RIG VISUALIZATION
# ============================================================

def draw_rig(
    image,
    parts
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
            index % len(colors)
        ]

        base = (
            int(part["base"][0]),
            int(part["base"][1])
        )

        tip = (
            int(part["tip"][0]),
            int(part["tip"][1])
        )

        center = (
            int(part["center"][0]),
            int(part["center"][1])
        )

        cv2.line(
            preview,
            base,
            tip,
            color,
            3
        )

        cv2.circle(
            preview,
            base,
            9,
            (0, 0, 255),
            -1
        )

        cv2.circle(
            preview,
            tip,
            7,
            (0, 255, 0),
            -1
        )

        cv2.circle(
            preview,
            center,
            5,
            color,
            -1
        )

        cv2.putText(
            preview,
            str(index + 1),
            (
                center[0] + 8,
                center[1] - 8
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            3,
            cv2.LINE_AA
        )

        cv2.putText(
            preview,
            str(index + 1),
            (
                center[0] + 8,
                center[1] - 8
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

    return preview


# ============================================================
# HTML GIF PREVIEW
# ============================================================

def gif_html(data):

    encoded = base64.b64encode(
        data
    ).decode("utf-8")

    return (
        '<img '
        f'src="data:image/gif;base64,{encoded}" '
        'style="width:100%;'
        'max-width:700px;'
        'border-radius:12px;'
        'display:block;'
        'margin:auto;">'
    )


# ============================================================
# MAIN
# ============================================================

uploaded = st.file_uploader(
    "Upload your hand-drawn character sheet",
    type=[
        "jpg",
        "jpeg",
        "png",
        "webp"
    ]
)


if uploaded is not None:

    try:

        # ----------------------------------------------------
        # LOAD
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
                "Could not read the image."
            )
            st.stop()

        # ----------------------------------------------------
        # SIDEBAR
        # ----------------------------------------------------

        st.sidebar.header(
            "⚙️ Image Settings"
        )

        rotation = st.sidebar.selectbox(
            "Rotate photograph",
            [0, 90, 180, 270],
            format_func=lambda x: (
                f"{x}°"
                if x != 0
                else "Original"
            )
        )

        image = rotate_image(
            image,
            rotation
        )

        h, w = image.shape[:2]

        if max(h, w) > MAX_IMAGE_SIZE:

            scale = (
                MAX_IMAGE_SIZE /
                max(h, w)
            )

            image = cv2.resize(
                image,
                (
                    int(w * scale),
                    int(h * scale)
                ),
                interpolation=cv2.INTER_AREA
            )

        # ----------------------------------------------------
        # COLORS
        # ----------------------------------------------------

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
                "No strong colored regions were detected."
            )

            st.image(
                cv2.cvtColor(
                    image,
                    cv2.COLOR_BGR2RGB
                ),
                width="stretch"
            )

            st.stop()

        color_dict = {
            c["label"]: c
            for c in detected_colors
        }

        default_colors = list(
            color_dict.keys()
        )[:2]

        selected_labels = (
            st.sidebar.multiselect(
                "Select colors that represent movable parts",
                list(color_dict.keys()),
                default=default_colors,
                help=(
                    "For your drawing, cyan and purple "
                    "are good candidates. Avoid selecting "
                    "paper/background colors."
                )
            )
        )

        selected_colors = [
            color_dict[label]
            for label in selected_labels
        ]

        # ----------------------------------------------------
        # MOTION
        # ----------------------------------------------------

        st.sidebar.header(
            "🎬 Animation"
        )

        motion = st.sidebar.selectbox(
            "Motion",
            MOTIONS
        )

        intensity = st.sidebar.slider(
            "Motion strength",
            2,
            40,
            14
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
                30
            ],
            value=12
        )

        duration_seconds = st.sidebar.slider(
            "Duration (seconds)",
            1.0,
            6.0,
            2.0,
            0.5
        )

        frame_count = max(
            8,
            int(
                fps *
                duration_seconds
            )
        )

        # ----------------------------------------------------
        # DETECT PARTS
        # ----------------------------------------------------

        parts = detect_parts(
            image,
            selected_colors
        )

        # ----------------------------------------------------
        # PREVIEW
        # ----------------------------------------------------

        col1, col2 = st.columns(2)

        with col1:

            st.subheader(
                "🖼️ Original Drawing"
            )

            st.image(
                cv2.cvtColor(
                    image,
                    cv2.COLOR_BGR2RGB
                ),
                width="stretch"
            )

        with col2:

            st.subheader(
                f"🦴 Detected Rig — {len(parts)} parts"
            )

            if parts:

                rig_preview = draw_rig(
                    image,
                    parts
                )

                st.image(
                    cv2.cvtColor(
                        rig_preview,
                        cv2.COLOR_BGR2RGB
                    ),
                    width="stretch"
                )

            else:

                st.info(
                    "Select one or more colored regions."
                )

        # ----------------------------------------------------
        # PART TABLE
        # ----------------------------------------------------

        if parts:

            with st.expander(
                "🔍 Detected Parts"
            ):

                for i, part in enumerate(parts):

                    parent = (
                        "ROOT"
                        if part["parent"] is None
                        else str(
                            part["parent"] + 1
                        )
                    )

                    st.write(
                        f"**Part {i + 1}** | "
                        f"{part['color_name']} | "
                        f"Area: {part['area']} | "
                        f"Parent: {parent} | "
                        f"Depth: {part['depth']}"
                    )

        st.markdown("---")

        # ----------------------------------------------------
        # GENERATE
        # ----------------------------------------------------

        generate = st.button(
            "✨ Generate Animation",
            type="primary",
            disabled=(len(parts) == 0),
            use_container_width=True
        )

        if generate:

            progress = st.progress(
                0,
                text="Preparing puppet rig..."
            )

            background = (
                prepare_background(
                    image,
                    parts
                )
            )

            frames = []

            for i in range(
                frame_count
            ):

                t = (
                    i /
                    max(
                        1,
                        frame_count - 1
                    )
                )

                # Smooth cyclical timing.
                t = (
                    t *
                    0.999
                )

                frame = render_frame(
                    image,
                    background,
                    parts,
                    motion,
                    t,
                    intensity
                )

                frames.append(
                    frame
                )

                progress.progress(
                    (i + 1) /
                    frame_count,
                    text=(
                        f"Rendering frame "
                        f"{i + 1}/{frame_count}"
                    )
                )

            progress.empty()

            # ------------------------------------------------
            # GIF
            # ------------------------------------------------

            gif_duration = int(
                1000 /
                fps
            )

            gif_data = build_gif(
                frames,
                gif_duration
            )

            st.success(
                f"🎉 {motion} animation generated!"
            )

            st.subheader(
                "🎬 Preview"
            )

            st.markdown(
                gif_html(gif_data),
                unsafe_allow_html=True
            )

            # ------------------------------------------------
            # DOWNLOADS
            # ------------------------------------------------

            d1, d2 = st.columns(2)

            safe_name = (
                motion
                .lower()
                .replace(" ", "_")
            )

            with d1:

                st.download_button(
                    "⬇️ Download GIF",
                    data=gif_data,
                    file_name=(
                        f"{safe_name}.gif"
                    ),
                    mime="image/gif",
                    use_container_width=True
                )

            with d2:

                with st.spinner(
                    "Encoding MP4..."
                ):

                    mp4_data = build_mp4(
                        frames,
                        fps
                    )

                if mp4_data:

                    st.download_button(
                        "🎞️ Download MP4",
                        data=mp4_data,
                        file_name=(
                            f"{safe_name}.mp4"
                        ),
                        mime="video/mp4",
                        use_container_width=True
                    )

                else:

                    st.info(
                        "MP4 encoding was unavailable. "
                        "GIF is still ready."
                    )

            # ------------------------------------------------
            # FRAME STRIP
            # ------------------------------------------------

            with st.expander(
                "🎞️ Animation Frames"
            ):

                frame_indices = np.linspace(
                    0,
                    len(frames) - 1,
                    min(
                        8,
                        len(frames)
                    )
                ).astype(int)

                cols = st.columns(
                    len(frame_indices)
                )

                for col, idx in zip(
                    cols,
                    frame_indices
                ):

                    with col:

                        st.image(
                            cv2.cvtColor(
                                frames[idx],
                                cv2.COLOR_BGR2RGB
                            ),
                            caption=f"Frame {idx + 1}"
                        )

    except Exception as e:

        st.error(
            f"❌ Error: {e}"
        )

        st.exception(e)

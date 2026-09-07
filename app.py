
import io
import json
import math
import os
import tempfile
import zipfile
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from google import genai
from google.genai import types


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Hand-Drawn Giraffe Walk Animator",
    page_icon="🦒",
    layout="wide",
)

st.title("🦒 Hand-Drawn Giraffe Walk Animator — Articulated v3")

st.caption(
    "White-canvas entrance → slow articulated walk → exact target position → "
    "smooth merge into the original drawing."
)


# ============================================================
# SETTINGS
# ============================================================

st.sidebar.header("⚙️ Animation Settings")

GEMINI_API_KEY = st.sidebar.text_input(
    "Gemini API Key",
    type="password",
)

ANIMATION_MODE = st.sidebar.selectbox(
    "Animation mode",
    [
        "White canvas → walk in → merge to original",
        "White canvas → walk in only",
        "Walk in place only",
    ],
)

TOTAL_FRAMES = st.sidebar.slider(
    "Total animation frames",
    32,
    160,
    80,
    4,
)

FPS = st.sidebar.slider(
    "FPS",
    4,
    20,
    6,
)

WALK_CYCLES = st.sidebar.slider(
    "Walking cycles before target",
    1,
    6,
    2,
)

st.sidebar.markdown("### 🦵 Giraffe Motion")

STEP_ANGLE = st.sidebar.slider(
    "Leg swing",
    3.0,
    25.0,
    11.0,
    0.5,
)

KNEE_BEND = st.sidebar.slider(
    "Knee bend",
    0.0,
    25.0,
    8.0,
    0.5,
)

FOOT_LIFT = st.sidebar.slider(
    "Foot lift",
    0.0,
    0.12,
    0.035,
    0.005,
)

BODY_BOB = st.sidebar.slider(
    "Body bob",
    0.0,
    0.04,
    0.006,
    0.001,
)

GROUND_LOCK = st.sidebar.slider(
    "Ground contact",
    0.0,
    1.0,
    0.90,
    0.05,
)

st.sidebar.markdown("### 🚶 Walk Timing")

WALK_IN_FRACTION = st.sidebar.slider(
    "Walk-in portion",
    0.15,
    0.65,
    0.38,
    0.02,
)

STAND_FRACTION = st.sidebar.slider(
    "Standing / settling portion",
    0.03,
    0.20,
    0.08,
    0.01,
)

MERGE_FRACTION = st.sidebar.slider(
    "Final merge portion",
    0.05,
    0.30,
    0.18,
    0.02,
)

st.sidebar.markdown("### 🧹 Character Extraction")

INK_DILATION = st.sidebar.slider(
    "Ink capture",
    1,
    8,
    3,
    1,
)

SHADOW_SUPPRESSION = st.sidebar.checkbox(
    "Suppress floor shadows",
    True,
)


# ============================================================
# BASIC HELPERS
# ============================================================

def ease(t: float) -> float:
    t = float(np.clip(t, 0.0, 1.0))
    return t * t * (3.0 - 2.0 * t)


def ease_in_out(t: float) -> float:
    t = float(np.clip(t, 0.0, 1.0))
    return 3 * t * t - 2 * t * t * t


def smootherstep(t: float) -> float:
    t = float(np.clip(t, 0.0, 1.0))
    return t * t * t * (t * (t * 6 - 15) + 10)


def clean_json_text(text: str) -> str:
    text = (text or "").strip()

    if text.startswith("```"):
        lines = text.splitlines()

        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]

        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        text = "\n".join(lines).strip()

    return text


# ============================================================
# GEMINI MODEL DISCOVERY
# ============================================================

def model_name(obj: Any) -> str:
    return str(
        getattr(obj, "name", "") or ""
    ).strip()


def model_actions(obj: Any) -> List[str]:

    a = getattr(
        obj,
        "supported_actions",
        None,
    )

    if a is None:
        a = getattr(
            obj,
            "supportedActions",
            None,
        )

    try:
        return [
            str(x)
            for x in (a or [])
        ]
    except Exception:
        return []


def discover_models(client: genai.Client) -> List[str]:

    models = []

    try:
        for m in client.models.list():

            name = model_name(m)

            if not name:
                continue

            actions = model_actions(m)

            if actions:
                if not any(
                    "generatecontent" in x.lower()
                    for x in actions
                ):
                    continue

            models.append(name)

    except Exception:
        return []

    unique = []
    seen = set()

    for name in models:

        key = name.lower()

        if key not in seen:
            seen.add(key)
            unique.append(name)

    return unique


def model_score(name: str) -> Tuple[int, str]:

    x = name.lower().replace(
        "models/",
        "",
    )

    # Prefer current Flash models if available.
    preferred = [
        ("gemini-3.7-flash", 0),
        ("gemini-3.6-flash", 1),
        ("gemini-3.5-flash", 2),
        ("gemini-3.1-flash", 3),
        ("gemini-3.1-flash-lite", 4),
        ("gemini-3", 5),
        ("gemini-2.5-flash", 6),
        ("gemini-2.5-flash-lite", 7),
        ("gemini-2.5", 8),
        ("gemini-2", 10),
        ("gemini", 20),
    ]

    for pattern, score in preferred:

        if pattern in x:
            return score, x

    return 100, x


def safe_text(resp: Any) -> str:

    text = getattr(
        resp,
        "text",
        None,
    )

    if text:
        return str(text)

    parts = []

    try:

        for candidate in (
            getattr(
                resp,
                "candidates",
                [],
            )
            or []
        ):

            content = getattr(
                candidate,
                "content",
                None,
            )

            for part in (
                getattr(
                    content,
                    "parts",
                    [],
                )
                or []
            ):

                if getattr(part, "text", None):
                    parts.append(
                        str(part.text)
                    )

    except Exception:
        pass

    return "\n".join(parts).strip()


# ============================================================
# GEMINI ANATOMY ANALYSIS
# ============================================================

def analyze_scene(
    image_bytes: bytes,
    api_key: str,
) -> Optional[Dict[str, Any]]:

    if not api_key:
        st.error(
            "Please enter your Gemini API key."
        )
        return None

    try:

        client = genai.Client(
            api_key=api_key
        )

    except Exception as e:

        st.error(
            f"Could not initialize Gemini: {e}"
        )
        return None

    models = discover_models(client)

    models = sorted(
        models,
        key=model_score,
    )

    if not models:

        st.error(
            "No generateContent-capable Gemini model "
            "was returned for this API key."
        )

        return None

    prompt = r"""
Analyze this SINGLE hand-drawn animal scene for a 2D cut-out
animation system.

IMPORTANT:
The Python program will NOT redraw the animal.

The Python program will move ORIGINAL pixels from the uploaded image.

Return ONLY geometry.

Coordinates must be normalized from 0..100 and written as [y,x].

The most important task is identifying the animal and its legs.

CRITICAL REQUIREMENTS:

1. Identify only the visible animal.
2. Ignore:
   - floor
   - table
   - cast shadow
   - glare
   - background
   - scenery
3. animal_polygon must tightly surround the visible animal.
4. For every clearly visible leg:
   - give a polygon tightly surrounding that leg
   - do not include floor shadow
   - do not invent hidden legs
5. Every leg must contain:
   proximal
   middle
   distal
6. proximal:
   exact visible attachment point to body.
7. middle:
   actual visible knee/elbow area.
   Do not simply place it halfway.
8. distal:
   visible ankle/wrist/hoof connection.
   Do not include cast shadow.
9. Give side when possible:
   front_left
   front_right
   back_left
   back_right
10. Give body polygon when clearly visible.
11. Give head, neck and tail polygons if clearly visible.
12. The animal_bbox must tightly contain the animal.
13. Estimate visible artwork only.
14. Do not imagine anatomy that cannot be seen.

The output must be valid JSON only.

Use this exact structure:

{
  "identified_character":"giraffe",
  "animal_bbox":[ymin,xmin,ymax,xmax],
  "animal_polygon":[[y,x],...],
  "parts":[
    {
      "name":"body",
      "type":"body",
      "polygon":[[y,x],...]
    },
    {
      "name":"front_left_leg",
      "type":"leg",
      "side":"front_left",
      "polygon":[[y,x],...],
      "joints":{
        "proximal":[y,x],
        "middle":[y,x],
        "distal":[y,x]
      }
    }
  ],
  "notes":"..."
}

If no animal is visible:

{
  "identified_character":"none",
  "animal_bbox":[0,0,0,0],
  "animal_polygon":[],
  "parts":[],
  "notes":"No visible animal"
}
"""

    errors = []

    for current_model in models:

        try:

            contents = [
                types.Part.from_bytes(
                    data=image_bytes,
                    mime_type="image/png",
                ),
                prompt,
            ]

            try:

                response = client.models.generate_content(
                    model=current_model,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        temperature=0.05,
                    ),
                )

            except Exception:

                response = client.models.generate_content(
                    model=current_model,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        temperature=0.05,
                    ),
                )

            text = safe_text(response)

            data = json.loads(
                clean_json_text(text)
            )

            if not isinstance(data, dict):
                raise ValueError(
                    "Gemini response was not a JSON object."
                )

            st.success(
                f"✅ Anatomy analysis completed with `{current_model}`"
            )

            return data

        except Exception as e:

            errors.append(
                f"{current_model}: "
                f"{str(e).replace(chr(10), ' ')}"
            )

    st.error(
        "Gemini analysis failed after trying "
        "all available models."
    )

    with st.expander("Model attempts"):

        for error in errors:
            st.code(error)

    return None


# ============================================================
# GEOMETRY
# ============================================================

def pt_px(
    p,
    w,
    h,
):
    y = float(
        np.clip(
            p[0],
            0,
            100,
        )
    )

    x = float(
        np.clip(
            p[1],
            0,
            100,
        )
    )

    return np.array(
        [
            x * w / 100.0,
            y * h / 100.0,
        ],
        dtype=np.float32,
    )


def poly_px(
    poly,
    w,
    h,
):

    if not isinstance(poly, list):
        return np.empty(
            (0, 2),
            np.int32,
        )

    points = []

    for p in poly:

        if (
            isinstance(p, (list, tuple))
            and len(p) >= 2
        ):

            points.append(
                pt_px(
                    p,
                    w,
                    h,
                )
            )

    if len(points) < 3:

        return np.empty(
            (0, 2),
            np.int32,
        )

    return np.asarray(
        points,
        np.int32,
    )


def poly_mask(
    shape,
    poly,
    dil=0,
):

    h, w = shape[:2]

    mask = np.zeros(
        (h, w),
        np.uint8,
    )

    points = poly_px(
        poly,
        w,
        h,
    )

    if len(points) >= 3:

        cv2.fillPoly(
            mask,
            [points],
            255,
        )

    if dil > 0:

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (
                2 * dil + 1,
                2 * dil + 1,
            ),
        )

        mask = cv2.dilate(
            mask,
            kernel,
        )

    return mask


def bbox_poly(
    poly,
    w,
    h,
    margin=10,
):

    p = poly_px(
        poly,
        w,
        h,
    )

    if len(p) < 3:
        return 0, 0, w, h

    x, y, bw, bh = cv2.boundingRect(p)

    return (
        max(0, x - margin),
        max(0, y - margin),
        min(w, x + bw + margin),
        min(h, y + bh + margin),
    )


def bbox_norm(
    bb,
    w,
    h,
    margin=10,
):

    if (
        not isinstance(bb, list)
        or len(bb) != 4
    ):

        return 0, 0, w, h

    ymin, xmin, ymax, xmax = [
        float(v)
        for v in bb
    ]

    x1 = int(
        xmin * w / 100
    )

    y1 = int(
        ymin * h / 100
    )

    x2 = int(
        xmax * w / 100
    )

    y2 = int(
        ymax * h / 100
    )

    return (
        max(0, x1 - margin),
        max(0, y1 - margin),
        min(w, x2 + margin),
        min(h, y2 + margin),
    )


def rotate_point(
    point,
    center,
    angle_deg,
):

    angle = math.radians(
        angle_deg
    )

    c = math.cos(angle)
    s = math.sin(angle)

    q = (
        np.asarray(
            point,
            dtype=np.float32,
        )
        - center
    )

    return center + np.array(
        [
            c * q[0] - s * q[1],
            s * q[0] + c * q[1],
        ],
        dtype=np.float32,
    )


def rotation_matrix(
    angle_deg,
    center,
):

    return cv2.getRotationMatrix2D(
        (
            float(center[0]),
            float(center[1]),
        ),
        angle_deg,
        1.0,
    )


# ============================================================
# FOREGROUND EXTRACTION
# ============================================================

def ink_mask(
    image_bgr,
    region_mask,
    shadow_suppress=True,
    dilation=3,
):

    hsv = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2HSV,
    )

    H, S, V = cv2.split(hsv)

    # Strong colored pixels.
    colored = (
        S > 38
    ).astype(
        np.uint8
    ) * 255

    strong_color = (
        (S > 55)
        & (V > 45)
    ).astype(
        np.uint8
    ) * 255

    support_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (
            2 * dilation + 5,
            2 * dilation + 5,
        ),
    )

    color_support = cv2.dilate(
        strong_color,
        support_kernel,
    )

    # Dark ink.
    dark = (
        V < 145
    ).astype(
        np.uint8
    ) * 255

    if shadow_suppress:

        # Black/brown outline is retained when it
        # is spatially close to actual colored artwork.
        dark_attached = cv2.bitwise_and(
            dark,
            color_support,
        )

        fg = cv2.bitwise_or(
            colored,
            dark_attached,
        )

    else:

        fg = cv2.bitwise_or(
            colored,
            dark,
        )

    fg = cv2.bitwise_and(
        fg,
        region_mask,
    )

    # Close tiny holes.
    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (3, 3),
    )

    fg = cv2.morphologyEx(
        fg,
        cv2.MORPH_CLOSE,
        close_kernel,
        iterations=1,
    )

    # Slight dilation to preserve hand-drawn edges.
    if dilation > 0:

        dil_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (
                2 * dilation + 1,
                2 * dilation + 1,
            ),
        )

        fg = cv2.dilate(
            fg,
            dil_kernel,
        )

    return fg


def rgba_from_crop(
    image_bgr,
    mask,
    bbox,
):

    x1, y1, x2, y2 = bbox

    crop = image_bgr[
        y1:y2,
        x1:x2,
    ].copy()

    crop_mask = mask[
        y1:y2,
        x1:x2,
    ]

    if crop.size == 0:

        return np.zeros(
            (1, 1, 4),
            np.uint8,
        )

    rgba = cv2.cvtColor(
        crop,
        cv2.COLOR_BGR2BGRA,
    )

    rgba[:, :, 3] = cv2.GaussianBlur(
        crop_mask,
        (3, 3),
        0,
    )

    return rgba


# ============================================================
# LEG EXTRACTION
# ============================================================

def prepare_leg(
    image_bgr,
    part,
):

    polygon = part.get(
        "polygon"
    ) or []

    joints = part.get(
        "joints"
    ) or {}

    required = (
        "proximal",
        "middle",
        "distal",
    )

    if (
        len(polygon) < 3
        or not all(
            key in joints
            for key in required
        )
    ):

        return None

    h, w = image_bgr.shape[:2]

    region = poly_mask(
        (h, w),
        polygon,
        dil=max(
            1,
            INK_DILATION,
        ),
    )

    mask = ink_mask(
        image_bgr,
        region,
        SHADOW_SUPPRESSION,
        INK_DILATION,
    )

    margin = max(
        25,
        min(h, w) // 45,
    )

    bbox = bbox_poly(
        polygon,
        w,
        h,
        margin=margin,
    )

    x1, y1, x2, y2 = bbox

    sprite = cv2.cvtColor(
        image_bgr[
            y1:y2,
            x1:x2
        ],
        cv2.COLOR_BGR2BGRA,
    )

    sprite_alpha = mask[
        y1:y2,
        x1:x2
    ]

    sprite[:, :, 3] = cv2.GaussianBlur(
        sprite_alpha,
        (3, 3),
        0,
    )

    global_joints = {
        key: pt_px(
            joints[key],
            w,
            h,
        )
        for key in required
    }

    local_joints = {
        key:
            global_joints[key]
            - np.array(
                [
                    x1,
                    y1,
                ],
                np.float32,
            )
        for key in required
    }

    return {
        "name": part.get(
            "name",
            "leg",
        ),
        "side": part.get(
            "side",
            part.get(
                "name",
                "leg",
            ),
        ),
        "sprite": sprite,
        "bbox": bbox,
        "joints": local_joints,
        "global": global_joints,
    }


def leg_parts(scene):

    result = []

    for part in (
        scene.get(
            "parts",
            []
        )
        or []
    ):

        if not isinstance(
            part,
            dict,
        ):
            continue

        part_type = str(
            part.get(
                "type",
                "",
            )
        ).lower()

        name = str(
            part.get(
                "name",
                "",
            )
        ).lower()

        if (
            part_type == "leg"
            or "leg" in name
        ):

            result.append(part)

    return result


# ============================================================
# AFFINE SPRITE UTILITIES
# ============================================================

def warp_rgba(
    sprite,
    alpha,
    matrix,
):

    h, w = sprite.shape[:2]

    warped = cv2.warpAffine(
        sprite,
        matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(
            0,
            0,
            0,
            0,
        ),
    )

    warped_alpha = cv2.warpAffine(
        alpha,
        matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    warped[:, :, 3] = warped_alpha

    return warped


def alpha_over(
    dst,
    src,
    x,
    y,
):

    if src is None or src.size == 0:
        return dst

    sh, sw = src.shape[:2]

    H, W = dst.shape[:2]

    x1 = max(
        0,
        int(x),
    )

    y1 = max(
        0,
        int(y),
    )

    x2 = min(
        W,
        int(x + sw),
    )

    y2 = min(
        H,
        int(y + sh),
    )

    if x1 >= x2 or y1 >= y2:
        return dst

    sx1 = x1 - int(x)
    sy1 = y1 - int(y)

    sx2 = sx1 + (x2 - x1)
    sy2 = sy1 + (y2 - y1)

    source = src[
        sy1:sy2,
        sx1:sx2
    ].astype(
        np.float32
    )

    alpha = (
        source[:, :, 3:4]
        / 255.0
    )

    target = dst[
        y1:y2,
        x1:x2
    ].astype(
        np.float32
    )

    result = (
        source[:, :, :3] * alpha
        + target * (1.0 - alpha)
    )

    dst[
        y1:y2,
        x1:x2
    ] = np.clip(
        result,
        0,
        255,
    ).astype(
        np.uint8
    )

    return dst


# ============================================================
# ARTICULATED LEG
# ============================================================

def articulated_leg(
    leg,
    swing,
    knee_bend,
    foot_lift,
    ground_lock,
    canvas_h,
):

    sprite = leg["sprite"]

    h, w = sprite.shape[:2]

    P = leg["joints"]["proximal"]
    M = leg["joints"]["middle"]
    D = leg["joints"]["distal"]

    # --------------------------------------------------------
    # Distance from proximal to distal.
    # Used to split upper and lower segments.
    # --------------------------------------------------------

    total_vec = D - P

    total_len = max(
        1.0,
        float(
            np.linalg.norm(
                total_vec
            )
        ),
    )

    unit = (
        total_vec
        / total_len
    )

    projection = np.dot(
        M - P,
        unit,
    )

    middle_ratio = float(
        np.clip(
            projection / total_len,
            0.20,
            0.80,
        )
    )

    # --------------------------------------------------------
    # Pixel coordinate grid.
    # --------------------------------------------------------

    yy, xx = np.mgrid[
        0:h,
        0:w
    ].astype(
        np.float32
    )

    points = np.stack(
        [
            xx,
            yy,
        ],
        axis=-1,
    )

    q = points - P

    distance_along = np.sum(
        q * unit,
        axis=-1,
    )

    middle_distance = (
        total_len
        * middle_ratio
    )

    # Soft transition around knee.
    transition = max(
        5.0,
        total_len * 0.08,
    )

    upper_weight = 1.0 - np.clip(
        (
            distance_along
            - middle_distance
        )
        / transition,
        0.0,
        1.0,
    )

    lower_weight = 1.0 - upper_weight

    base_alpha = (
        sprite[:, :, 3]
        .astype(np.float32)
        / 255.0
    )

    upper_alpha = np.clip(
        base_alpha
        * upper_weight,
        0.0,
        1.0,
    ) * 255

    lower_alpha = np.clip(
        base_alpha
        * lower_weight,
        0.0,
        1.0,
    ) * 255

    upper_alpha = (
        upper_alpha
        .astype(np.uint8)
    )

    lower_alpha = (
        lower_alpha
        .astype(np.uint8)
    )

    # --------------------------------------------------------
    # UPPER LEG
    #
    # Rotates around proximal attachment.
    # --------------------------------------------------------

    upper_matrix = rotation_matrix(
        swing,
        P,
    )

    upper = warp_rgba(
        sprite,
        upper_alpha,
        upper_matrix,
    )

    # Where the knee has moved.
    moved_knee = rotate_point(
        M,
        P,
        swing,
    )

    # --------------------------------------------------------
    # LOWER LEG
    #
    # Its original orientation receives:
    #
    # body swing
    # +
    # knee articulation
    #
    # This creates a real bend.
    # --------------------------------------------------------

    lower_absolute_angle = (
        swing
        + knee_bend
    )

    lower_matrix = rotation_matrix(
        lower_absolute_angle,
        M,
    )

    lower = warp_rgba(
        sprite,
        lower_alpha,
        lower_matrix,
    )

    # The rotation around original M keeps M fixed.
    # Now move the entire lower segment so its knee
    # coincides with the newly moved knee.
    shift = (
        moved_knee
        - M
    )

    translation = np.array(
        [
            [1, 0, shift[0]],
            [0, 1, shift[1]],
        ],
        dtype=np.float32,
    )

    lower = cv2.warpAffine(
        lower,
        translation,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(
            0,
            0,
            0,
            0,
        ),
    )

    # --------------------------------------------------------
    # FOOT LIFT
    #
    # Image y decreases upward.
    # Ground lock prevents excessive floating.
    # --------------------------------------------------------

    effective_lift = (
        foot_lift
        * (1.0 - 0.70 * ground_lock)
    )

    lift_pixels = (
        effective_lift
        * canvas_h
    )

    if lift_pixels > 0.01:

        lift_matrix = np.array(
            [
                [1, 0, 0],
                [0, 1, -lift_pixels],
            ],
            dtype=np.float32,
        )

        lower = cv2.warpAffine(
            lower,
            lift_matrix,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(
                0,
                0,
                0,
                0,
            ),
        )

    return upper, lower


# ============================================================
# CLEAN ORIGINAL CHARACTER
# ============================================================

def extract_animal(
    image,
    scene,
):

    h, w = image.shape[:2]

    polygon = (
        scene.get(
            "animal_polygon"
        )
        or []
    )

    if len(polygon) >= 3:

        bbox = bbox_poly(
            polygon,
            w,
            h,
            margin=max(
                10,
                min(h, w) // 150,
            ),
        )

        region = poly_mask(
            (h, w),
            polygon,
            dil=INK_DILATION,
        )

    else:

        bbox = bbox_norm(
            scene.get(
                "animal_bbox"
            ),
            w,
            h,
            margin=max(
                10,
                min(h, w) // 150,
            ),
        )

        region = np.zeros(
            (h, w),
            np.uint8,
        )

        x1, y1, x2, y2 = bbox

        region[
            y1:y2,
            x1:x2
        ] = 255

    mask = ink_mask(
        image,
        region,
        SHADOW_SUPPRESSION,
        INK_DILATION,
    )

    x1, y1, x2, y2 = bbox

    rgba = cv2.cvtColor(
        image[
            y1:y2,
            x1:x2
        ],
        cv2.COLOR_BGR2BGRA,
    )

    rgba[:, :, 3] = mask[
        y1:y2,
        x1:x2
    ]

    return rgba, bbox, mask


# ============================================================
# REMOVE CHARACTER FROM ORIGINAL
# ============================================================

def remove_animal_from_original(
    image,
    animal,
    bbox,
):

    result = image.copy()

    x1, y1, x2, y2 = bbox

    mask = np.zeros(
        image.shape[:2],
        np.uint8,
    )

    alpha = animal[
        :,
        :,
        3
    ]

    hh = min(
        alpha.shape[0],
        y2 - y1,
    )

    ww = min(
        alpha.shape[1],
        x2 - x1,
    )

    mask[
        y1:y1 + hh,
        x1:x1 + ww
    ] = alpha[
        :hh,
        :ww
    ]

    mask = cv2.dilate(
        mask,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (9, 9),
        ),
    )

    try:

        result = cv2.inpaint(
            result,
            mask,
            8,
            cv2.INPAINT_TELEA,
        )

    except Exception:
        pass

    return result


# ============================================================
# ANIMAL PLATE
# ============================================================

def clean_animation_plate(
    image,
    scene,
):

    plate = image.copy()

    for part in (
        scene.get(
            "parts",
            []
        )
        or []
    ):

        if not isinstance(
            part,
            dict,
        ):
            continue

        is_leg = (
            str(
                part.get(
                    "type",
                    "",
                )
            ).lower()
            == "leg"
            or
            "leg" in str(
                part.get(
                    "name",
                    "",
                )
            ).lower()
        )

        if not is_leg:
            continue

        polygon = (
            part.get(
                "polygon"
            )
            or []
        )

        if len(polygon) < 3:
            continue

        region = poly_mask(
            plate.shape,
            polygon,
            dil=max(
                3,
                INK_DILATION + 2,
            ),
        )

        mask = ink_mask(
            plate,
            region,
            SHADOW_SUPPRESSION,
            INK_DILATION + 1,
        )

        mask = cv2.dilate(
            mask,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (9, 9),
            ),
        )

        try:

            plate = cv2.inpaint(
                plate,
                mask,
                6,
                cv2.INPAINT_TELEA,
            )

        except Exception:
            pass

    return plate


# ============================================================
# GAIT PHASE
# ============================================================

def leg_phase(
    side,
    index,
):

    s = str(
        side
    ).lower()

    # Natural diagonal gait.
    #
    # Front-left + back-right
    # move together.
    #
    # Front-right + back-left
    # move together.

    if "front_left" in s:
        return 0.0

    if "back_right" in s:
        return 0.0

    if "front_right" in s:
        return math.pi

    if "back_left" in s:
        return math.pi

    return (
        0.0
        if index % 2 == 0
        else math.pi
    )


# ============================================================
# SINGLE WALKING POSE
# ============================================================

def render_walk_pose(
    plate,
    prepared_legs,
    phase,
    body_offset_y=0,
):

    frame = plate.copy()

    h, w = plate.shape[:2]

    for index, leg in enumerate(
        prepared_legs
    ):

        phase_offset = leg_phase(
            leg["side"],
            index,
        )

        s = math.sin(
            phase
            + phase_offset
        )

        # Slightly softer than a perfect sine.
        gait = (
            0.72 * s
            + 0.28 * s * abs(s)
        )

        side = str(
            leg["side"]
        ).lower()

        swing = (
            gait
            * STEP_ANGLE
        )

        if "back" in side:
            swing *= 0.92

        # Lift happens primarily during forward swing.
        forward = max(
            0.0,
            gait,
        )

        lift = (
            forward
            * FOOT_LIFT
        )

        # Knee flexion.
        #
        # When the leg moves forward,
        # bend the knee slightly.
        if gait > 0:

            knee = -(
                KNEE_BEND
                * gait
            )

        else:

            knee = (
                KNEE_BEND
                * 0.35
                * gait
            )

        upper, lower = articulated_leg(
            leg,
            swing,
            knee,
            lift,
            GROUND_LOCK,
            h,
        )

        bx, by, _, _ = leg["bbox"]

        draw_y = (
            by
            + int(body_offset_y)
        )

        frame = alpha_over(
            frame,
            upper,
            bx,
            draw_y,
        )

        frame = alpha_over(
            frame,
            lower,
            bx,
            draw_y,
        )

    return frame


# ============================================================
# WALKING KEYFRAMES
# ============================================================

def create_walk_keyframes(
    image,
    prepared_legs,
    scene,
):

    plate = clean_animation_plate(
        image,
        scene,
    )

    # 8 poses instead of 4.
    #
    # This is important for a more natural
    # giraffe gait.

    phases = np.linspace(
        0,
        2 * math.pi,
        9,
        endpoint=False,
    )

    frames = []

    for phase in phases:

        bob = (
            math.sin(
                phase * 2
            )
            * image.shape[0]
            * BODY_BOB
        )

        frame = render_walk_pose(
            plate,
            prepared_legs,
            phase,
            bob,
        )

        frames.append(frame)

    return frames


# ============================================================
# STANDING FRAME
# ============================================================

def create_standing_frame(
    image,
    prepared_legs,
    scene,
):

    plate = clean_animation_plate(
        image,
        scene,
    )

    # Legs nearly neutral.
    frame = render_walk_pose(
        plate,
        prepared_legs,
        0.0,
        0,
    )

    return frame


# ============================================================
# WHITE CANVAS
# ============================================================

def white_canvas(
    image,
):

    h, w = image.shape[:2]

    return np.full(
        (h, w, 3),
        255,
        dtype=np.uint8,
    )


# ============================================================
# PLACE ANIMAL
# ============================================================

def place_animal_at_target(
    canvas,
    animal,
    bbox,
    progress,
    from_left=True,
):

    x1, y1, x2, y2 = bbox

    ah, aw = animal.shape[:2]

    # Exact target location.
    target_x = x1
    target_y = y1

    # Start completely outside left edge.
    start_x = -aw - 40

    if from_left:

        p = smootherstep(
            progress
        )

        current_x = int(
            start_x
            + (
                target_x
                - start_x
            )
            * p
        )

    else:

        current_x = target_x

    current_y = target_y

    return alpha_over(
        canvas,
        animal,
        current_x,
        current_y,
    )


# ============================================================
# MOVE WALKING ANIMAL
# ============================================================

def create_walk_in_frame(
    white,
    walking_pose,
    animal_bbox,
    animal_position_progress,
):

    x1, y1, x2, y2 = animal_bbox

    ph, pw = walking_pose.shape[:2]

    start_x = -pw - 50

    target_x = x1

    p = smootherstep(
        animal_position_progress
    )

    x = int(
        start_x
        + (
            target_x
            - start_x
        )
        * p
    )

    # Very small vertical body movement.
    #
    # Avoid excessive bouncing because a giraffe
    # has a relatively stable torso while walking.
    y = y1

    return alpha_over(
        white.copy(),
        walking_pose,
        x,
        y,
    )


# ============================================================
# EXACT TARGET POSITION
# ============================================================

def create_target_frame(
    white,
    walking_pose,
    bbox,
):

    x1, y1, _, _ = bbox

    return alpha_over(
        white.copy(),
        walking_pose,
        x1,
        y1,
    )


# ============================================================
# SMOOTH MERGE
# ============================================================

def smooth_merge(
    character_frame,
    original,
    p,
):

    p = smootherstep(
        p
    )

    return cv2.addWeighted(
        character_frame,
        1.0 - p,
        original,
        p,
        0,
    )


# ============================================================
# BUILD COMPLETE ANIMATION
# ============================================================

def build_animation(
    image,
    animal,
    bbox,
    walk_keys,
    total_frames,
    cycles,
    mode,
    walk_fraction,
    stand_fraction,
    merge_fraction,
):

    original = image.copy()

    white = white_canvas(
        image
    )

    if not walk_keys:
        return [
            original.copy()
            for _ in range(total_frames)
        ]

    # --------------------------------------------------------
    # FRAME ALLOCATION
    # --------------------------------------------------------

    if mode == "Walk in place only":

        walk_in_n = 0
        stand_n = 0
        merge_n = 0

    else:

        walk_in_n = max(
            8,
            int(
                total_frames
                * walk_fraction
            ),
        )

        stand_n = max(
            3,
            int(
                total_frames
                * stand_fraction
            ),
        )

        if mode == "White canvas → walk in only":

            merge_n = 0

        else:

            merge_n = max(
                4,
                int(
                    total_frames
                    * merge_fraction
                ),
            )

    remaining = (
        total_frames
        - walk_in_n
        - stand_n
        - merge_n
    )

    if remaining < 1:

        remaining = 1

    # --------------------------------------------------------
    # WALK CYCLE FRAMES
    # --------------------------------------------------------

    cycle_count = max(
        1,
        int(cycles),
    )

    # Use all 8 keyframes.
    cycle_length = len(
        walk_keys
    )

    walking_frames = []

    for i in range(
        max(
            remaining,
            cycle_count * cycle_length,
        )
    ):

        idx = (
            i
            % cycle_length
        )

        next_idx = (
            idx + 1
        ) % cycle_length

        # Smooth interpolation between
        # actual articulated drawings.
        local_t = (
            i
            % cycle_length
        ) / float(
            cycle_length
        )

        # Slight ease.
        blend = smootherstep(
            local_t
        )

        frame = cv2.addWeighted(
            walk_keys[idx],
            1.0 - blend,
            walk_keys[next_idx],
            blend,
            0,
        )

        walking_frames.append(
            frame
        )

    seq = []

    # --------------------------------------------------------
    # PHASE 1:
    # PURE WHITE CANVAS
    # GIRAFFE WALKS IN FROM LEFT
    # --------------------------------------------------------

    if walk_in_n > 0:

        for i in range(
            walk_in_n
        ):

            if walk_in_n == 1:
                p = 1.0
            else:
                p = (
                    i
                    / float(
                        walk_in_n - 1
                    )
                )

            # Make the gait slightly slower.
            cycle_progress = (
                p
                * max(
                    1.0,
                    cycle_count
                )
            )

            phase_index = (
                cycle_progress
                * cycle_length
            )

            idx = int(
                phase_index
            ) % cycle_length

            next_idx = (
                idx + 1
            ) % cycle_length

            blend = smootherstep(
                phase_index
                - int(
                    phase_index
                )
            )

            pose = cv2.addWeighted(
                walk_keys[idx],
                1.0 - blend,
                walk_keys[next_idx],
                blend,
                0,
            )

            frame = create_walk_in_frame(
                white,
                pose,
                bbox,
                p,
            )

            seq.append(
                frame
            )

    # --------------------------------------------------------
    # PHASE 2:
    # GIRAFFE REACHES EXACT LOCATION
    # --------------------------------------------------------

    if stand_n > 0:

        last_pose = walk_keys[
            0
        ]

        # Use several almost-still frames.
        # This gives the feeling of arriving
        # and settling its weight.
        for i in range(
            stand_n
        ):

            if stand_n == 1:
                p = 1.0
            else:
                p = (
                    i
                    / float(
                        stand_n - 1
                    )
                )

            # Tiny residual body movement.
            settling = (
                math.sin(
                    p * math.pi
                )
                * image.shape[0]
                * BODY_BOB
                * 0.35
            )

            target = create_target_frame(
                white,
                last_pose,
                bbox,
            )

            if abs(
                settling
            ) > 0.01:

                # Apply tiny vertical movement
                # only to character.
                x1, y1, x2, y2 = bbox

                shifted = np.full_like(
                    target,
                    255,
                )

                shifted = alpha_over(
                    shifted,
                    last_pose,
                    x1,
                    y1 + int(settling),
                )

                target = shifted

            seq.append(
                target
            )

    # --------------------------------------------------------
    # PHASE 3:
    # OPTIONAL WALK-IN-PLACE
    # --------------------------------------------------------

    remaining_walk = max(
        0,
        remaining,
    )

    for i in range(
        remaining_walk
    ):

        idx = (
            i
            % len(
                walking_frames
            )
        )

        frame = walking_frames[
            idx
        ].copy()

        # Walking frames are on the original
        # coordinate system, so the giraffe is
        # already exactly at its target.
        seq.append(
            frame
        )

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # If the requested animation is meant to
    # finish with the original image, reserve
    # the final merge frames.
    #
    # We rebuild the sequence so merge always
    # occurs at the very end.
    # --------------------------------------------------------

    if merge_n > 0:

        # Make sure we have a target-position
        # character frame.
        target_pose = walk_keys[
            0
        ]

        character_target = create_target_frame(
            white,
            target_pose,
            bbox,
        )

        # Replace any accidental excess frames.
        if len(seq) > total_frames - merge_n:

            seq = seq[
                :total_frames - merge_n
            ]

        # Add merge.
        for i in range(
            merge_n
        ):

            if merge_n == 1:
                p = 1.0
            else:
                p = (
                    i
                    / float(
                        merge_n - 1
                    )
                )

            merged = smooth_merge(
                character_target,
                original,
                p,
            )

            seq.append(
                merged
            )

    # --------------------------------------------------------
    # WALK-IN-PLACE ONLY
    # --------------------------------------------------------

    if mode == "Walk in place only":

        seq = []

        for i in range(
            total_frames
        ):

            idx = (
                i
                % len(
                    walking_frames
                )
            )

            seq.append(
                walking_frames[idx].copy()
            )

    # --------------------------------------------------------
    # FINAL LENGTH
    # --------------------------------------------------------

    if len(seq) > total_frames:

        seq = seq[
            :total_frames
        ]

    while len(seq) < total_frames:

        if mode == "White canvas → walk in → merge":

            seq.append(
                original.copy()
            )

        else:

            seq.append(
                seq[-1].copy()
                if seq
                else white.copy()
            )

    # Guarantee final original frame
    # when merge mode is active.
    if mode == "White canvas → walk in → merge":

        seq[-1] = original.copy()

    return seq


# ============================================================
# DETECTION VISUALIZATION
# ============================================================

def detection_overlay(
    image,
    scene,
):

    output = image.copy()

    h, w = output.shape[:2]

    animal_polygon = poly_px(
        scene.get(
            "animal_polygon"
        )
        or [],
        w,
        h,
    )

    if len(
        animal_polygon
    ) >= 3:

        cv2.polylines(
            output,
            [
                animal_polygon
            ],
            True,
            (0, 180, 0),
            max(
                2,
                min(h, w) // 300,
            ),
        )

    for part in (
        scene.get(
            "parts",
            []
        )
        or []
    ):

        if not isinstance(
            part,
            dict,
        ):
            continue

        polygon = poly_px(
            part.get(
                "polygon"
            )
            or [],
            w,
            h,
        )

        if len(
            polygon
        ) >= 3:

            cv2.polylines(
                output,
                [
                    polygon
                ],
                True,
                (255, 120, 0),
                max(
                    1,
                    min(h, w) // 450,
                ),
            )

        joints = part.get(
            "joints",
            {}
        ) or {}

        for name, joint in joints.items():

            if (
                isinstance(
                    joint,
                    (list, tuple)
                )
                and len(joint) >= 2
            ):

                x, y = (
                    pt_px(
                        joint,
                        w,
                        h,
                    )
                    .astype(int)
                )

                cv2.circle(
                    output,
                    (x, y),
                    6,
                    (0, 0, 255),
                    -1,
                )

                cv2.putText(
                    output,
                    str(name),
                    (x + 7, y - 7),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 0, 255),
                    1,
                    cv2.LINE_AA,
                )

    return output


# ============================================================
# EXPORTS
# ============================================================

def gif_bytes(
    frames,
    fps,
):

    if not frames:
        return b""

    images = [
        Image.fromarray(
            cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB,
            )
        )
        for frame in frames
    ]

    buffer = io.BytesIO()

    images[0].save(
        buffer,
        format="GIF",
        save_all=True,
        append_images=images[1:],
        duration=max(
            30,
            int(
                1000 / fps
            ),
        ),
        loop=0,
        optimize=False,
    )

    return buffer.getvalue()


def mp4_bytes(
    frames,
    fps,
):

    if not frames:
        return None

    fd, path = tempfile.mkstemp(
        suffix=".mp4"
    )

    os.close(fd)

    h, w = frames[0].shape[:2]

    writer = cv2.VideoWriter(
        path,
        cv2.VideoWriter_fourcc(
            *"mp4v"
        ),
        float(fps),
        (w, h),
    )

    if not writer.isOpened():

        try:
            os.remove(path)
        except OSError:
            pass

        return None

    for frame in frames:
        writer.write(frame)

    writer.release()

    try:

        with open(
            path,
            "rb",
        ) as file:

            return file.read()

    finally:

        try:
            os.remove(path)
        except OSError:
            pass


def zip_frames(
    frames,
    prefix="frame",
):

    buffer = io.BytesIO()

    with zipfile.ZipFile(
        buffer,
        "w",
        zipfile.ZIP_DEFLATED,
    ) as archive:

        for i, frame in enumerate(
            frames,
            1,
        ):

            ok, encoded = cv2.imencode(
                ".png",
                frame,
            )

            if ok:

                archive.writestr(
                    f"{prefix}_{i:03d}.png",
                    encoded.tobytes(),
                )

    return buffer.getvalue()


# ============================================================
# USER INTERFACE
# ============================================================

uploaded = st.file_uploader(
    "Upload your hand-drawn giraffe scene",
    type=[
        "png",
        "jpg",
        "jpeg",
    ],
)


if not uploaded:

    st.info(
        "Upload the original drawing containing the giraffe "
        "and its background."
    )

    st.stop()


raw = uploaded.getvalue()


# ============================================================
# LOAD IMAGE
# ============================================================

try:

    pil = Image.open(
        io.BytesIO(raw)
    ).convert("RGB")

    png_buffer = io.BytesIO()

    pil.save(
        png_buffer,
        format="PNG",
    )

    gemini_bytes = (
        png_buffer.getvalue()
    )

except Exception as e:

    st.error(
        f"Could not prepare image: {e}"
    )

    st.stop()


image = cv2.imdecode(
    np.frombuffer(
        raw,
        np.uint8,
    ),
    cv2.IMREAD_COLOR,
)


if image is None:

    st.error(
        "Could not read image."
    )

    st.stop()


# ============================================================
# ORIGINAL PREVIEW
# ============================================================

c1, c2 = st.columns(2)

with c1:

    st.image(
        cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB,
        ),
        caption="Original drawing",
        use_container_width=True,
    )


with c2:

    st.markdown(
        """
### 🦒 New animation sequence

**1. White canvas**

The animation begins with a completely white canvas.

**2. Giraffe enters**

Only the giraffe is visible.
The background does not move.

**3. Slow articulated walk**

The legs use:

- hip/shoulder joint
- knee joint
- ankle/hoof joint

**4. Exact target position**

The giraffe moves until its bounding box reaches the
same position as the giraffe in the original drawing.

**5. Settling**

The giraffe briefly stands at that exact position.

**6. Smooth reconstruction**

The original drawing gradually appears.

**7. Final frame**

The result becomes exactly the uploaded original image.
"""
    )


# ============================================================
# ANALYZE
# ============================================================

if st.button(
    "🔍 Analyze drawing with Gemini",
    type="primary",
    use_container_width=True,
):

    with st.spinner(
        "Analyzing giraffe anatomy..."
    ):

        scene_result = analyze_scene(
            gemini_bytes,
            GEMINI_API_KEY,
        )

    if scene_result:

        st.session_state.scene = (
            scene_result
        )

        st.session_state.pop(
            "walk_keyframes",
            None,
        )

        st.session_state.pop(
            "frames",
            None,
        )


if "scene" not in st.session_state:

    st.stop()


scene = st.session_state.scene


st.success(
    f"Detected: **{scene.get('identified_character', 'character')}**"
)


with st.expander(
    "Gemini anatomy JSON"
):

    st.json(scene)


# ============================================================
# DETECTION PREVIEW
# ============================================================

st.image(
    cv2.cvtColor(
        detection_overlay(
            image,
            scene,
        ),
        cv2.COLOR_BGR2RGB,
    ),
    caption=(
        "Green = animal | Orange = legs | "
        "Red = joints"
    ),
    use_container_width=True,
)


# ============================================================
# PREPARE LEGS
# ============================================================

parts = leg_parts(
    scene
)

prepared = []

for part in parts:

    prepared_leg = prepare_leg(
        image,
        part,
    )

    if prepared_leg is not None:

        prepared.append(
            prepared_leg
        )


if not prepared:

    st.error(
        "No usable leg polygons/joints were returned. "
        "Try a clearer image or analyze again."
    )

    st.stop()


st.success(
    f"Prepared {len(prepared)} original leg cut-outs."
)


# ============================================================
# STEP 2
# ============================================================

st.markdown("---")

st.header(
    "🦵 Step 2 — Generate slow articulated walking poses"
)

st.write(
    "The giraffe is no longer treated as a rigid object. "
    "Each visible leg bends around its detected knee while "
    "the upper section follows the body attachment."
)


if st.button(
    "🦒 Generate 8 improved walking poses",
    type="primary",
    use_container_width=True,
):

    with st.spinner(
        "Building articulated giraffe gait..."
    ):

        keyframes = create_walk_keyframes(
            image,
            prepared,
            scene,
        )

        st.session_state[
            "walk_keyframes"
        ] = keyframes

        st.session_state.pop(
            "frames",
            None,
        )


if "walk_keyframes" in st.session_state:

    keyframes = st.session_state[
        "walk_keyframes"
    ]

    cols = st.columns(4)

    for i, frame in enumerate(
        keyframes
    ):

        with cols[
            i % 4
        ]:

            st.image(
                cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB,
                ),
                caption=f"Pose {i + 1}",
                use_container_width=True,
            )

    st.download_button(
        "⬇️ Download 8 key poses",
        zip_frames(
            keyframes,
            "giraffe_walk_pose",
        ),
        "giraffe_walk_keyposes.zip",
        "application/zip",
        use_container_width=True,
    )


# ============================================================
# STEP 3
# ============================================================

st.markdown("---")

st.header(
    "🎞️ Step 3 — Build white-canvas entrance animation"
)


if "walk_keyframes" not in st.session_state:

    st.info(
        "First generate the walking poses above."
    )

else:

    if st.button(
        "🚀 Render complete animation",
        type="primary",
        use_container_width=True,
    ):

        with st.spinner(
            "Rendering slow giraffe entrance and merge..."
        ):

            animal, bbox, animal_mask = extract_animal(
                image,
                scene,
            )

            frames = build_animation(
                image=image,
                animal=animal,
                bbox=bbox,
                walk_keys=st.session_state[
                    "walk_keyframes"
                ],
                total_frames=TOTAL_FRAMES,
                cycles=WALK_CYCLES,
                mode=ANIMATION_MODE,
                walk_fraction=WALK_IN_FRACTION,
                stand_fraction=STAND_FRACTION,
                merge_fraction=MERGE_FRACTION,
            )

            st.session_state[
                "frames"
            ] = frames

        st.success(
            f"Rendered {len(frames)} frames at {FPS} FPS."
        )


# ============================================================
# RESULT
# ============================================================

if "frames" in st.session_state:

    frames = st.session_state[
        "frames"
    ]

    gif = gif_bytes(
        frames,
        FPS,
    )

    st.markdown(
        "### 🎉 Final animation"
    )

    st.image(
        gif,
        caption=(
            "White canvas → slow giraffe entrance → "
            "exact target position → original scene"
        ),
        use_container_width=True,
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.download_button(
            "⬇️ Download GIF",
            gif,
            "giraffe_white_canvas_walk.gif",
            "image/gif",
            use_container_width=True,
        )

    with col2:

        mp4 = mp4_bytes(
            frames,
            FPS,
        )

        if mp4:

            st.download_button(
                "⬇️ Download MP4",
                mp4,
                "giraffe_white_canvas_walk.mp4",
                "video/mp4",
                use_container_width=True,
            )

        else:

            st.info(
                "MP4 unavailable in this environment."
            )

    with col3:

        st.download_button(
            "⬇️ Download PNG frames",
            zip_frames(
                frames,
                "giraffe_frame",
            ),
            "giraffe_animation_frames.zip",
            "application/zip",
            use_container_width=True,
        )


    # ========================================================
    # FRAME PREVIEW
    # ========================================================

    st.markdown(
        "### 🖼️ Animation stages"
    )

    preview_indices = [
        0,
        int(len(frames) * 0.10),
        int(len(frames) * 0.25),
        int(len(frames) * 0.40),
        int(len(frames) * 0.55),
        int(len(frames) * 0.70),
        int(len(frames) * 0.85),
        len(frames) - 1,
    ]

    preview_indices = sorted(
        set(
            np.clip(
                preview_indices,
                0,
                len(frames) - 1,
            ).astype(int)
        )
    )

    cols = st.columns(4)

    for n, idx in enumerate(
        preview_indices
    ):

        with cols[
            n % 4
        ]:

            st.image(
                cv2.cvtColor(
                    frames[idx],
                    cv2.COLOR_BGR2RGB,
                ),
                caption=f"Frame {idx + 1}",
                use_container_width=True,
            )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "v3 — Original-pixel articulated animation. "
    "Gemini supplies anatomy geometry only. "
    "No AI redraw is used for the animation artwork."
)

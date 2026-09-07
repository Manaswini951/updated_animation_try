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
    page_title="Hand-Drawn Animal Walk Animator",
    page_icon="🦒",
    layout="wide",
)

st.title("🦒 Hand-Drawn Animal Walk Animator — Connected Walk v4")

st.caption(
    "Gemini analyzes anatomy only. Python moves the ORIGINAL drawing pixels. "
    "No AI redraw is used."
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
        "White canvas → walk in → stand → merge",
        "White canvas → walk in → stand",
        "Walk in place only",
    ],
)

TOTAL_FRAMES = st.sidebar.slider(
    "Total frames",
    30,
    160,
    80,
    2,
)

FPS = st.sidebar.slider(
    "FPS",
    4,
    20,
    7,
)

WALK_CYCLES = st.sidebar.slider(
    "Walking cycles after entering",
    0,
    5,
    2,
)

st.sidebar.markdown("### 🦵 Giraffe movement")

STEP_ANGLE = st.sidebar.slider(
    "Leg swing",
    2.0,
    18.0,
    8.0,
    0.5,
)

KNEE_BEND = st.sidebar.slider(
    "Knee bend",
    0.0,
    16.0,
    5.0,
    0.5,
)

FOOT_LIFT = st.sidebar.slider(
    "Foot lift",
    0.0,
    0.08,
    0.025,
    0.005,
)

BODY_BOB = st.sidebar.slider(
    "Body bob",
    0.0,
    0.025,
    0.004,
    0.001,
)

GROUND_LOCK = st.sidebar.slider(
    "Ground contact",
    0.0,
    1.0,
    0.90,
    0.05,
)

st.sidebar.markdown("### 🎬 Timing")

WALK_IN_FRACTION = st.sidebar.slider(
    "Walk-in portion",
    0.10,
    0.50,
    0.25,
    0.02,
)

STAND_FRACTION = st.sidebar.slider(
    "Standing / walking-in-place portion",
    0.10,
    0.60,
    0.45,
    0.02,
)

MERGE_FRACTION = st.sidebar.slider(
    "Final scenery merge",
    0.05,
    0.35,
    0.18,
    0.02,
)

st.sidebar.markdown("### 🧹 Drawing extraction")

INK_DILATION = st.sidebar.slider(
    "Ink capture",
    1,
    7,
    3,
    1,
)

SHADOW_SUPPRESSION = st.sidebar.checkbox(
    "Suppress floor shadows",
    True,
)

st.sidebar.markdown("### 🚶 Entry")

ENTRY_SIDE = st.sidebar.selectbox(
    "Animal enters from",
    ["Left", "Right"],
)

ENTRY_EXTRA_DISTANCE = st.sidebar.slider(
    "Entry distance",
    0.0,
    0.30,
    0.08,
    0.01,
)


# ============================================================
# GEMINI
# ============================================================

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


def model_name(obj: Any) -> str:
    return str(
        getattr(obj, "name", "") or ""
    ).strip()


def model_actions(obj: Any) -> List[str]:
    actions = getattr(obj, "supported_actions", None)

    if actions is None:
        actions = getattr(obj, "supportedActions", None)

    try:
        return [str(x) for x in (actions or [])]
    except Exception:
        return []


def discover_models(client: genai.Client) -> List[str]:

    models = []

    try:
        for model in client.models.list():

            name = model_name(model)

            if not name:
                continue

            actions = model_actions(model)

            if actions:
                if not any(
                    "generatecontent" in x.lower()
                    for x in actions
                ):
                    continue

            models.append(name)

    except Exception:
        return []

    seen = set()
    result = []

    for name in models:

        key = name.lower()

        if key not in seen:
            seen.add(key)
            result.append(name)

    return result


def model_score(name: str) -> Tuple[int, str]:

    x = name.lower().replace("models/", "")

    patterns = [
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

    for pattern, score in patterns:

        if pattern in x:
            return score, x

    return 100, x


def safe_text(response: Any) -> str:

    text = getattr(response, "text", None)

    if text:
        return str(text)

    parts = []

    try:

        for candidate in getattr(
            response,
            "candidates",
            [],
        ) or []:

            content = getattr(
                candidate,
                "content",
                None,
            )

            for part in getattr(
                content,
                "parts",
                [],
            ) or []:

                value = getattr(
                    part,
                    "text",
                    None,
                )

                if value:
                    parts.append(str(value))

    except Exception:
        pass

    return "\n".join(parts).strip()


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

    except Exception as exc:

        st.error(
            f"Could not initialize Gemini: {exc}"
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
Analyze this SINGLE hand-drawn animal scene for a
2D cut-out animation system.

DO NOT redraw the animal.

Return geometry only.

Coordinates must be normalized 0..100
and represented as [y,x].

The Python program will move ORIGINAL pixels.

IMPORTANT:

1. Identify ONLY the main visible animal.

2. The animal polygon must surround the visible
   animal but exclude background and floor shadows.

3. Identify EVERY clearly visible leg.

4. Do NOT invent hidden legs.

5. Each leg must have a tight polygon around
   the actual visible leg.

6. Do NOT include floor shadows.

7. Each leg must contain:

   proximal
   middle
   distal

8. PROXIMAL:
   exact point where the visible leg attaches
   to the body.

9. MIDDLE:
   actual knee/elbow/bend location.

10. DISTAL:
    ankle/wrist/hoof area before any shadow.

11. For a giraffe, preserve the very long
    characteristic leg geometry.

12. Give a side whenever possible:

    front_left
    front_right
    back_left
    back_right

13. Also identify body/head/neck/tail when visible.

14. Be conservative.
    Follow the actual ink.

Return ONLY valid JSON.

Schema:

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
 "animal_bbox":[],
 "animal_polygon":[],
 "parts":[],
 "notes":"..."
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

                response = (
                    client.models.generate_content(
                        model=current_model,
                        contents=contents,
                        config=types.GenerateContentConfig(
                            response_mime_type="application/json",
                            temperature=0.05,
                        ),
                    )
                )

            except Exception:

                response = (
                    client.models.generate_content(
                        model=current_model,
                        contents=contents,
                        config=types.GenerateContentConfig(
                            temperature=0.05,
                        ),
                    )
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
                f"✅ Anatomy analyzed with `{current_model}`"
            )

            return data

        except Exception as exc:

            errors.append(
                f"{current_model}: "
                f"{str(exc).replace(chr(10), ' ')}"
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
    width,
    height,
):
    y = float(
        np.clip(p[0], 0, 100)
    )

    x = float(
        np.clip(p[1], 0, 100)
    )

    return np.array(
        [
            x * width / 100.0,
            y * height / 100.0,
        ],
        dtype=np.float32,
    )


def poly_px(
    polygon,
    width,
    height,
):

    if not isinstance(
        polygon,
        list,
    ):
        return np.empty(
            (0, 2),
            dtype=np.int32,
        )

    points = []

    for p in polygon:

        if (
            isinstance(p, (list, tuple))
            and len(p) >= 2
        ):
            points.append(
                pt_px(
                    p,
                    width,
                    height,
                )
            )

    if len(points) < 3:

        return np.empty(
            (0, 2),
            dtype=np.int32,
        )

    return np.asarray(
        points,
        dtype=np.int32,
    )


def poly_mask(
    shape,
    polygon,
    dilation=0,
):

    height, width = shape[:2]

    mask = np.zeros(
        (height, width),
        dtype=np.uint8,
    )

    points = poly_px(
        polygon,
        width,
        height,
    )

    if len(points) >= 3:

        cv2.fillPoly(
            mask,
            [points],
            255,
        )

    if dilation > 0:

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (
                2 * dilation + 1,
                2 * dilation + 1,
            ),
        )

        mask = cv2.dilate(
            mask,
            kernel,
        )

    return mask


def bbox_poly(
    polygon,
    width,
    height,
    margin=10,
):

    points = poly_px(
        polygon,
        width,
        height,
    )

    if len(points) < 3:

        return (
            0,
            0,
            width,
            height,
        )

    x, y, bw, bh = cv2.boundingRect(
        points
    )

    return (
        max(0, x - margin),
        max(0, y - margin),
        min(width, x + bw + margin),
        min(height, y + bh + margin),
    )


def bbox_norm(
    bbox,
    width,
    height,
    margin=10,
):

    if (
        not isinstance(bbox, list)
        or len(bbox) != 4
    ):
        return (
            0,
            0,
            width,
            height,
        )

    ymin, xmin, ymax, xmax = [
        float(v)
        for v in bbox
    ]

    x1 = int(
        xmin * width / 100
    )

    y1 = int(
        ymin * height / 100
    )

    x2 = int(
        xmax * width / 100
    )

    y2 = int(
        ymax * height / 100
    )

    return (
        max(0, x1 - margin),
        max(0, y1 - margin),
        min(width, x2 + margin),
        min(height, y2 + margin),
    )


def rotation_matrix(
    angle,
    center,
):

    return cv2.getRotationMatrix2D(
        (
            float(center[0]),
            float(center[1]),
        ),
        float(angle),
        1.0,
    )


def rotate_point(
    point,
    center,
    angle,
):

    angle_rad = math.radians(
        float(angle)
    )

    c = math.cos(angle_rad)
    s = math.sin(angle_rad)

    q = (
        np.asarray(
            point,
            dtype=np.float32,
        )
        - center
    )

    return (
        center
        + np.array(
            [
                c * q[0] - s * q[1],
                s * q[0] + c * q[1],
            ],
            dtype=np.float32,
        )
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

    colored = (
        S > 35
    ).astype(np.uint8) * 255

    strong_color = (
        (S > 55)
        & (V > 45)
    ).astype(np.uint8) * 255

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

    dark = (
        V < 145
    ).astype(np.uint8) * 255

    dark_attached = cv2.bitwise_and(
        dark,
        color_support,
    )

    if shadow_suppress:
        foreground = cv2.bitwise_or(
            colored,
            dark_attached,
        )
    else:
        foreground = cv2.bitwise_or(
            colored,
            dark,
        )

    foreground = cv2.bitwise_and(
        foreground,
        region_mask,
    )

    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (3, 3),
    )

    foreground = cv2.morphologyEx(
        foreground,
        cv2.MORPH_CLOSE,
        close_kernel,
        iterations=1,
    )

    if dilation > 0:

        foreground = cv2.dilate(
            foreground,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (
                    2 * dilation + 1,
                    2 * dilation + 1,
                ),
            ),
        )

    return foreground


# ============================================================
# SAFE ALPHA COMPOSITING
# ============================================================

def alpha_over(
    destination,
    source,
    x,
    y,
):
    """
    VERY defensive alpha compositor.

    Fixes the previous broadcasting error by ensuring:
      - source is BGRA
      - alpha is H x W x 1
      - destination is BGR
      - clipping is done before multiplication
    """

    if source is None:
        return destination

    if source.size == 0:
        return destination

    destination = np.asarray(
        destination,
        dtype=np.uint8,
    )

    source = np.asarray(
        source,
        dtype=np.uint8,
    )

    if source.ndim != 3:
        return destination

    if source.shape[2] == 3:

        alpha_channel = np.full(
            source.shape[:2],
            255,
            dtype=np.uint8,
        )

        source = np.dstack(
            [
                source,
                alpha_channel,
            ]
        )

    if source.shape[2] < 4:
        return destination

    H, W = destination.shape[:2]

    sh, sw = source.shape[:2]

    x = int(x)
    y = int(y)

    x1 = max(0, x)
    y1 = max(0, y)

    x2 = min(W, x + sw)
    y2 = min(H, y + sh)

    if x1 >= x2 or y1 >= y2:
        return destination

    sx1 = x1 - x
    sy1 = y1 - y

    sx2 = sx1 + (x2 - x1)
    sy2 = sy1 + (y2 - y1)

    src = source[
        sy1:sy2,
        sx1:sx2,
    ]

    src_rgb = src[:, :, :3].astype(
        np.float32
    )

    alpha = src[:, :, 3].astype(
        np.float32
    ) / 255.0

    if alpha.ndim == 2:
        alpha = alpha[:, :, None]

    dst = destination[
        y1:y2,
        x1:x2,
    ].astype(np.float32)

    if dst.ndim == 2:

        dst = cv2.cvtColor(
            dst.astype(np.uint8),
            cv2.COLOR_GRAY2BGR,
        ).astype(np.float32)

    if dst.shape[2] != 3:
        dst = dst[:, :, :3]

    result = (
        src_rgb * alpha
        + dst * (1.0 - alpha)
    )

    destination[
        y1:y2,
        x1:x2,
    ] = np.clip(
        result,
        0,
        255,
    ).astype(np.uint8)

    return destination


# ============================================================
# RGBA TRANSFORM HELPERS
# ============================================================

def warp_rgba(
    sprite,
    alpha_mask,
    matrix,
):

    h, w = sprite.shape[:2]

    transformed = cv2.warpAffine(
        sprite,
        matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )

    transformed_alpha = cv2.warpAffine(
        alpha_mask,
        matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    transformed[:, :, 3] = transformed_alpha

    return transformed


def translate_rgba(
    sprite,
    dx,
    dy,
):

    h, w = sprite.shape[:2]

    matrix = np.array(
        [
            [1, 0, float(dx)],
            [0, 1, float(dy)],
        ],
        dtype=np.float32,
    )

    return cv2.warpAffine(
        sprite,
        matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )


# ============================================================
# LEG DETECTION / PREPARATION
# ============================================================

def leg_parts(scene):

    result = []

    for part in scene.get(
        "parts",
        [],
    ) or []:

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

    if (
        len(polygon) < 3
        or not all(
            key in joints
            for key in (
                "proximal",
                "middle",
                "distal",
            )
        )
    ):
        return None

    h, w = image_bgr.shape[:2]

    region = poly_mask(
        (h, w),
        polygon,
        dilation=max(
            2,
            INK_DILATION,
        ),
    )

    mask = ink_mask(
        image_bgr,
        region,
        SHADOW_SUPPRESSION,
        INK_DILATION,
    )

    # Bigger margin is important because
    # the leg will rotate.
    margin = max(
        30,
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

    local_mask = mask[
        y1:y2,
        x1:x2
    ].copy()

    # Slightly strengthen thin hand-drawn lines.
    local_mask = cv2.dilate(
        local_mask,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (3, 3),
        ),
    )

    sprite[:, :, 3] = cv2.GaussianBlur(
        local_mask,
        (3, 3),
        0,
    )

    global_joints = {
        key: pt_px(
            joints[key],
            w,
            h,
        )
        for key in (
            "proximal",
            "middle",
            "distal",
        )
    }

    local_joints = {
        key: global_joints[key]
        - np.array(
            [x1, y1],
            dtype=np.float32,
        )
        for key in global_joints
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
        "mask": local_mask,
    }


# ============================================================
# CONNECTED ARTICULATED LEG
# ============================================================

def leg_root_radius(leg):

    P = leg["joints"]["proximal"]
    M = leg["joints"]["middle"]

    length = float(
        np.linalg.norm(M - P)
    )

    return max(
        5,
        min(
            24,
            int(length * 0.18),
        ),
    )


def articulated_leg(
    leg,
    swing,
    knee,
    foot_lift,
):

    """
    IMPORTANT DIFFERENCE FROM OLD VERSION:

    The leg is NOT split at a hard boundary.

    Instead:

        BODY
          │
       ROOT/COLLAR
          │
      ========   <- overlap
          │
        KNEE
       /    \
      /      \
     LOWER LEG

    The proximal root remains attached to the body.
    Upper and lower sections overlap at the knee.
    """

    sprite = leg["sprite"]

    h, w = sprite.shape[:2]

    P = leg["joints"]["proximal"]
    M = leg["joints"]["middle"]
    D = leg["joints"]["distal"]

    # --------------------------------------------------------
    # Segment directions
    # --------------------------------------------------------

    upper_vector = M - P
    lower_vector = D - M

    upper_length = max(
        1.0,
        float(
            np.linalg.norm(
                upper_vector
            )
        ),
    )

    lower_length = max(
        1.0,
        float(
            np.linalg.norm(
                lower_vector
            )
        ),
    )

    upper_unit = (
        upper_vector
        / upper_length
    )

    lower_unit = (
        lower_vector
        / lower_length
    )

    root_radius = leg_root_radius(
        leg
    )

    knee_overlap = max(
        5,
        min(
            18,
            int(
                min(
                    upper_length,
                    lower_length,
                )
                * 0.15
            ),
        ),
    )

    # --------------------------------------------------------
    # Pixel coordinate grid
    # --------------------------------------------------------

    yy, xx = np.mgrid[
        0:h,
        0:w,
    ].astype(np.float32)

    q = np.stack(
        [xx, yy],
        axis=-1,
    )

    alpha = sprite[
        :,
        :,
        3
    ]

    # --------------------------------------------------------
    # Projection along upper leg
    # --------------------------------------------------------

    upper_projection = np.sum(
        (q - P) * upper_unit,
        axis=-1,
    )

    # --------------------------------------------------------
    # Projection along lower leg
    # --------------------------------------------------------

    lower_projection = np.sum(
        (q - M) * lower_unit,
        axis=-1,
    )

    base_alpha = alpha.astype(
        np.float32
    )

    # --------------------------------------------------------
    # Upper segment
    #
    # Start BEFORE root radius so that it
    # overlaps the body collar.
    # --------------------------------------------------------

    upper_region = (
        (upper_projection >= root_radius * 0.35)
        &
        (
            upper_projection
            <= upper_length
            + knee_overlap
        )
    )

    upper_mask = np.where(
        upper_region,
        base_alpha,
        0,
    ).astype(np.uint8)

    # --------------------------------------------------------
    # Lower segment
    #
    # Starts BEFORE knee so upper/lower
    # physically overlap.
    # --------------------------------------------------------

    lower_region = (
        (lower_projection >= -knee_overlap)
        &
        (
            lower_projection
            <= lower_length
            + knee_overlap
        )
    )

    lower_mask = np.where(
        lower_region,
        base_alpha,
        0,
    ).astype(np.uint8)

    # Extra overlap.
    upper_mask = cv2.dilate(
        upper_mask,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (3, 3),
        ),
    )

    lower_mask = cv2.dilate(
        lower_mask,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (3, 3),
        ),
    )

    # --------------------------------------------------------
    # UPPER LEG
    # --------------------------------------------------------

    upper_matrix = rotation_matrix(
        swing,
        P,
    )

    upper = warp_rgba(
        sprite,
        upper_mask,
        upper_matrix,
    )

    # Exact moved knee.
    M1 = rotate_point(
        M,
        P,
        swing,
    )

    # --------------------------------------------------------
    # FOOT LIFT
    #
    # Do NOT translate the lower leg vertically.
    #
    # Translation was one of the reasons the
    # old version broke the knee connection.
    #
    # Instead add a small rotation around
    # the knee.
    # --------------------------------------------------------

    lift_pixels = (
        float(foot_lift)
        * h
    )

    lift_angle = 0.0

    if lift_pixels > 0:

        lift_angle = math.degrees(
            math.atan2(
                lift_pixels * 0.30,
                max(
                    lower_length,
                    1.0,
                ),
            )
        )

        lift_angle = min(
            lift_angle,
            7.0,
        )

        # Normal walking leg:
        # foot is below knee -> rotate clockwise
        # to raise it slightly.
        if D[1] >= M[1]:
            lift_angle = -lift_angle

    # Lower leg absolute orientation.
    lower_angle = (
        swing
        + knee
        + lift_angle
    )

    # Rotate around ORIGINAL knee.
    lower_rotated = warp_rgba(
        sprite,
        lower_mask,
        rotation_matrix(
            lower_angle,
            M,
        ),
    )

    # Move ORIGINAL knee to MOVED knee.
    lower = translate_rgba(
        lower_rotated,
        M1[0] - M[0],
        M1[1] - M[1],
    )

    return upper, lower


# ============================================================
# ANIMAL EXTRACTION
# ============================================================

def extract_animal(
    image,
    scene,
):

    h, w = image.shape[:2]

    polygon = scene.get(
        "animal_polygon"
    ) or []

    if len(polygon) >= 3:

        bbox = bbox_poly(
            polygon,
            w,
            h,
            margin=max(
                15,
                min(h, w) // 100,
            ),
        )

        region = poly_mask(
            (h, w),
            polygon,
            dilation=INK_DILATION,
        )

    else:

        bbox = bbox_norm(
            scene.get(
                "animal_bbox"
            ),
            w,
            h,
            margin=max(
                15,
                min(h, w) // 100,
            ),
        )

        region = np.zeros(
            (h, w),
            dtype=np.uint8,
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

    crop = image[
        y1:y2,
        x1:x2
    ]

    rgba = cv2.cvtColor(
        crop,
        cv2.COLOR_BGR2BGRA,
    )

    rgba[:, :, 3] = cv2.GaussianBlur(
        mask[
            y1:y2,
            x1:x2
        ],
        (3, 3),
        0,
    )

    return (
        rgba,
        bbox,
        mask,
    )


# ============================================================
# REMOVE ANIMAL FROM ORIGINAL BACKGROUND
# ============================================================

def remove_animal_from_background(
    image,
    animal,
    bbox,
):

    background = image.copy()

    x1, y1, x2, y2 = bbox

    if (
        x2 <= x1
        or y2 <= y1
    ):
        return background

    alpha = animal[
        :,
        :,
        3
    ]

    full_mask = np.zeros(
        image.shape[:2],
        dtype=np.uint8,
    )

    full_mask[
        y1:y2,
        x1:x2
    ] = alpha

    full_mask = cv2.dilate(
        full_mask,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (9, 9),
        ),
    )

    try:

        return cv2.inpaint(
            background,
            full_mask,
            7,
            cv2.INPAINT_TELEA,
        )

    except Exception:

        return background


# ============================================================
# BODY-ONLY SPRITE
# ============================================================

def make_body_sprite(
    animal,
    animal_bbox,
    prepared_legs,
):

    body = animal.copy()

    ax1, ay1, ax2, ay2 = animal_bbox

    body_h, body_w = body.shape[:2]

    for leg in prepared_legs:

        lx1, ly1, lx2, ly2 = leg[
            "bbox"
        ]

        local_x1 = lx1 - ax1
        local_y1 = ly1 - ay1
        local_x2 = lx2 - ax1
        local_y2 = ly2 - ay1

        # Clip.
        bx1 = max(
            0,
            local_x1,
        )

        by1 = max(
            0,
            local_y1,
        )

        bx2 = min(
            body_w,
            local_x2,
        )

        by2 = min(
            body_h,
            local_y2,
        )

        if (
            bx1 >= bx2
            or by1 >= by2
        ):
            continue

        sx1 = bx1 - local_x1
        sy1 = by1 - local_y1

        sx2 = sx1 + (
            bx2 - bx1
        )

        sy2 = sy1 + (
            by2 - by1
        )

        leg_mask = leg[
            "mask"
        ][
            sy1:sy2,
            sx1:sx2
        ].copy()

        # ----------------------------------------------------
        # CRITICAL:
        # Keep a collar of ORIGINAL pixels around
        # the proximal attachment point.
        # ----------------------------------------------------

        P = leg[
            "joints"
        ]["proximal"]

        root_radius = leg_root_radius(
            leg
        )

        px = int(
            P[0] - (
                lx1 - ax1
            )
        )

        py = int(
            P[1] - (
                ly1 - ay1
            )
        )

        px -= sx1
        py -= sy1

        cv2.circle(
            leg_mask,
            (
                px,
                py,
            ),
            root_radius,
            0,
            -1,
        )

        body_alpha = body[
            by1:by2,
            bx1:bx2,
            3
        ]

        body_alpha[
            leg_mask > 0
        ] = 0

        body[
            by1:by2,
            bx1:bx2,
            3
        ] = body_alpha

    return body


# ============================================================
# WALKING PHASE
# ============================================================

def phase_for(
    side,
    index,
):

    s = str(
        side
    ).lower()

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


def smooth_gait(value):

    # Smooth sinusoidal movement.
    return (
        value
        * (
            0.72
            + 0.28
            * abs(value)
        )
    )


# ============================================================
# WALKING POSE
# ============================================================

def create_walking_pose(
    background,
    body_sprite,
    body_bbox,
    prepared_legs,
    t,
):

    frame = background.copy()

    h, w = background.shape[:2]

    # --------------------------------------------------------
    # Gentle whole-body bob.
    #
    # IMPORTANT:
    # Body and ALL legs receive exactly the same bob.
    # This prevents the old "leg detached from body" problem.
    # --------------------------------------------------------

    bob = int(
        math.sin(
            2.0
            * math.pi
            * t
        )
        * h
        * BODY_BOB
    )

    bx, by, _, _ = body_bbox

    # Body moves together with legs.
    frame = alpha_over(
        frame,
        body_sprite,
        bx,
        by + bob,
    )

    # --------------------------------------------------------
    # Legs
    # --------------------------------------------------------

    for index, leg in enumerate(
        prepared_legs
    ):

        phase = phase_for(
            leg["side"],
            index,
        )

        raw = math.sin(
            2.0
            * math.pi
            * t
            + phase
        )

        gait = smooth_gait(
            raw
        )

        side = str(
            leg["side"]
        ).lower()

        swing = (
            gait
            * STEP_ANGLE
        )

        if "back" in side:
            swing *= 0.90

        # Feet lift only during positive phase.
        lift = max(
            0.0,
            gait,
        ) * FOOT_LIFT

        lift *= (
            1.0
            - 0.65
            * GROUND_LOCK
        )

        # Knee bends only during swing.
        if raw > 0:

            knee = (
                -KNEE_BEND
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
        )

        lx, ly, _, _ = leg[
            "bbox"
        ]

        # SAME bob as body.
        ly += bob

        # Upper first.
        frame = alpha_over(
            frame,
            upper,
            lx,
            ly,
        )

        # Lower second.
        # Overlap makes the knee visually connected.
        frame = alpha_over(
            frame,
            lower,
            lx,
            ly,
        )

    return frame


# ============================================================
# STANDING POSE
# ============================================================

def create_standing_pose(
    background,
    body_sprite,
    body_bbox,
    prepared_legs,
):

    frame = background.copy()

    bx, by, _, _ = body_bbox

    frame = alpha_over(
        frame,
        body_sprite,
        bx,
        by,
    )

    for leg in prepared_legs:

        upper, lower = articulated_leg(
            leg,
            0.0,
            0.0,
            0.0,
        )

        lx, ly, _, _ = leg[
            "bbox"
        ]

        frame = alpha_over(
            frame,
            upper,
            lx,
            ly,
        )

        frame = alpha_over(
            frame,
            lower,
            lx,
            ly,
        )

    return frame


# ============================================================
# WALK-IN
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
        t
        * t
        * (
            3.0
            - 2.0
            * t
        )
    )


def create_walk_in_frame(
    white_canvas,
    animal,
    target_bbox,
    p,
):

    H, W = white_canvas.shape[:2]

    target_x1, target_y1, target_x2, target_y2 = (
        target_bbox
    )

    animal_h, animal_w = (
        animal.shape[:2]
    )

    p = ease_in_out(
        p
    )

    target_x = target_x1

    target_y = target_y1

    # Extra distance outside canvas.
    extra = int(
        W
        * ENTRY_EXTRA_DISTANCE
    )

    if ENTRY_SIDE == "Left":

        start_x = (
            -animal_w
            - extra
        )

    else:

        start_x = (
            W
            + extra
        )

    x = int(
        start_x
        + (
            target_x
            - start_x
        )
        * p
    )

    # Very subtle walking bob while entering.
    y = int(
        target_y
        + math.sin(
            p
            * math.pi
            * 2.0
        )
        * H
        * 0.004
    )

    frame = white_canvas.copy()

    return alpha_over(
        frame,
        animal,
        x,
        y,
    )


# ============================================================
# FINAL MERGE
# ============================================================

def smooth_merge(
    animated_frame,
    original,
    p,
):

    p = ease_in_out(
        p
    )

    # Slightly smoother than direct linear dissolve.
    return cv2.addWeighted(
        animated_frame,
        1.0 - p,
        original,
        p,
        0,
    )


# ============================================================
# COMPLETE ANIMATION
# ============================================================

def build_animation(
    image,
    animal,
    animal_bbox,
    body_sprite,
    prepared_legs,
    total_frames,
    walk_cycles,
    mode,
    walk_in_fraction,
    stand_fraction,
    merge_fraction,
):

    H, W = image.shape[:2]

    # --------------------------------------------------------
    # PURE WHITE INITIAL CANVAS
    # --------------------------------------------------------

    white = np.full(
        (H, W, 3),
        255,
        dtype=np.uint8,
    )

    # --------------------------------------------------------
    # Background used after animal is removed.
    #
    # It is NOT shown during the walk-in.
    # --------------------------------------------------------

    clean_background = (
        remove_animal_from_background(
            image,
            animal,
            animal_bbox,
        )
    )

    # --------------------------------------------------------
    # Determine frame counts.
    # --------------------------------------------------------

    if mode == "Walk in place only":

        walk_n = 0
        merge_n = 0

        stand_n = total_frames

    else:

        walk_n = max(
            8,
            int(
                total_frames
                * walk_in_fraction
            ),
        )

        if mode == "White canvas → walk in → stand":

            merge_n = 0

        else:

            merge_n = max(
                5,
                int(
                    total_frames
                    * merge_fraction
                ),
            )

        stand_n = max(
            8,
            total_frames
            - walk_n
            - merge_n,
        )

    frames = []

    # ========================================================
    # PHASE 1
    # PURE WHITE + ANIMAL ENTERS
    # ========================================================

    if walk_n > 0:

        for i in range(
            walk_n
        ):

            if walk_n == 1:
                p = 1.0
            else:
                p = i / (
                    walk_n - 1
                )

            frame = create_walk_in_frame(
                white,
                animal,
                animal_bbox,
                p,
            )

            frames.append(
                frame
            )

    # ========================================================
    # PHASE 2
    # ANIMAL IS NOW AT EXACT ORIGINAL POSITION
    #
    # It walks gently in place.
    # ========================================================

    if stand_n > 0:

        if walk_cycles <= 0:

            for _ in range(
                stand_n
            ):

                frame = create_standing_pose(
                    clean_background,
                    body_sprite,
                    animal_bbox,
                    prepared_legs,
                )

                frames.append(
                    frame
                )

        else:

            for i in range(
                stand_n
            ):

                # Number of complete cycles.
                t = (
                    i
                    / max(
                        1,
                        stand_n,
                    )
                    * walk_cycles
                )

                frame = create_walking_pose(
                    clean_background,
                    body_sprite,
                    animal_bbox,
                    prepared_legs,
                    t,
                )

                frames.append(
                    frame
                )

    # ========================================================
    # PHASE 3
    # SETTLE INTO EXACT ORIGINAL POSE
    # ========================================================

    if merge_n > 0:

        if frames:

            last_frame = frames[-1]

        else:

            last_frame = white.copy()

        # First few merge frames remain animated,
        # then dissolve into the original scenery.
        for i in range(
            merge_n
        ):

            if merge_n == 1:
                p = 1.0
            else:
                p = i / (
                    merge_n - 1
                )

            frame = smooth_merge(
                last_frame,
                image,
                p,
            )

            frames.append(
                frame
            )

    # --------------------------------------------------------
    # Exact frame count.
    # --------------------------------------------------------

    if len(frames) > total_frames:

        frames = frames[
            :total_frames
        ]

    while len(frames) < total_frames:

        frames.append(
            image.copy()
            if merge_n > 0
            else (
                frames[-1].copy()
                if frames
                else white.copy()
            )
        )

    # Guarantee final frame = ORIGINAL.
    if (
        mode
        == "White canvas → walk in → stand → merge"
    ):
        frames[-1] = image.copy()

    return frames


# ============================================================
# EXPORT
# ============================================================

def gif_bytes(
    frames,
    fps,
):

    if not frames:
        return b""

    images = []

    for frame in frames:

        frame = np.asarray(
            frame,
            dtype=np.uint8,
        )

        if frame.ndim == 2:

            frame = cv2.cvtColor(
                frame,
                cv2.COLOR_GRAY2RGB,
            )

        else:

            frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB,
            )

        images.append(
            Image.fromarray(
                frame
            )
        )

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
        except Exception:
            pass

        return None

    for frame in frames:

        writer.write(
            frame
        )

    writer.release()

    try:

        with open(
            path,
            "rb",
        ) as f:

            return f.read()

    finally:

        try:
            os.remove(path)
        except Exception:
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
        ) or [],
        w,
        h,
    )

    if len(animal_polygon) >= 3:

        cv2.polylines(
            output,
            [animal_polygon],
            True,
            (0, 180, 0),
            max(
                2,
                min(h, w) // 300,
            ),
        )

    for part in scene.get(
        "parts",
        [],
    ) or []:

        if not isinstance(
            part,
            dict,
        ):
            continue

        polygon = poly_px(
            part.get(
                "polygon"
            ) or [],
            w,
            h,
        )

        if len(polygon) >= 3:

            cv2.polylines(
                output,
                [polygon],
                True,
                (255, 120, 0),
                max(
                    1,
                    min(h, w) // 450,
                ),
            )

        joints = part.get(
            "joints"
        ) or {}

        for name, joint in joints.items():

            if (
                isinstance(
                    joint,
                    (list, tuple),
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
                    (
                        x + 6,
                        y - 6,
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 0, 255),
                    1,
                    cv2.LINE_AA,
                )

    return output


# ============================================================
# UI
# ============================================================

uploaded = st.file_uploader(
    "Upload your hand-drawn animal scene",
    type=[
        "png",
        "jpg",
        "jpeg",
    ],
)

if not uploaded:

    st.info(
        "Upload the drawing containing the giraffe/animal "
        "and its original background."
    )

    st.stop()


raw = uploaded.getvalue()


# ============================================================
# READ IMAGE
# ============================================================

try:

    pil_image = Image.open(
        io.BytesIO(raw)
    ).convert("RGB")

    png_buffer = io.BytesIO()

    pil_image.save(
        png_buffer,
        format="PNG",
    )

    gemini_bytes = (
        png_buffer.getvalue()
    )

except Exception as exc:

    st.error(
        f"Could not prepare image: {exc}"
    )

    st.stop()


image = cv2.imdecode(
    np.frombuffer(
        raw,
        dtype=np.uint8,
    ),
    cv2.IMREAD_COLOR,
)

if image is None:

    st.error(
        "Could not read image."
    )

    st.stop()


# ============================================================
# ORIGINAL IMAGE
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
### 🎬 New animation sequence

**1. Completely white canvas**

⬇️

**2. Animal enters from the side**

⬇️

**3. Animal reaches the exact position where it originally stood**

⬇️

**4. Animal walks gently in place**

⬇️

**5. Legs remain connected to the body**

⬇️

**6. Animal settles**

⬇️

**7. Original scenery smoothly appears**

The background is intentionally **not shown during the
walk-in**.
"""
    )


# ============================================================
# ANALYSIS
# ============================================================

if st.button(
    "🔍 Analyze drawing with Gemini",
    type="primary",
    use_container_width=True,
):

    with st.spinner(
        "Gemini is analyzing the animal anatomy..."
    ):

        scene = analyze_scene(
            gemini_bytes,
            GEMINI_API_KEY,
        )

    if scene:

        st.session_state[
            "scene"
        ] = scene

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


scene = st.session_state[
    "scene"
]

st.success(
    f"Detected: **{scene.get('identified_character', 'animal')}**"
)


with st.expander(
    "Gemini anatomy JSON"
):

    st.json(
        scene
    )


st.image(
    cv2.cvtColor(
        detection_overlay(
            image,
            scene,
        ),
        cv2.COLOR_BGR2RGB,
    ),
    caption=(
        "Green = animal | "
        "Orange = legs | "
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

prepared_legs = []

for part in parts:

    prepared = prepare_leg(
        image,
        part,
    )

    if prepared is not None:

        prepared_legs.append(
            prepared
        )


if not prepared_legs:

    st.error(
        "Gemini did not return usable leg "
        "polygons and joints."
    )

    st.stop()


st.success(
    f"Prepared {len(prepared_legs)} connected "
    f"original-pixel leg cut-outs."
)


# ============================================================
# EXTRACT ANIMAL
# ============================================================

animal, animal_bbox, animal_mask = (
    extract_animal(
        image,
        scene,
    )
)


# ============================================================
# BODY SPRITE
# ============================================================

body_sprite = make_body_sprite(
    animal,
    animal_bbox,
    prepared_legs,
)


# ============================================================
# DEBUG / EXTRACTION PREVIEW
# ============================================================

with st.expander(
    "🔬 Extraction preview"
):

    preview_white = np.full(
        image.shape,
        255,
        dtype=np.uint8,
    )

    preview_white = alpha_over(
        preview_white,
        animal,
        animal_bbox[0],
        animal_bbox[1],
    )

    st.image(
        cv2.cvtColor(
            preview_white,
            cv2.COLOR_BGR2RGB,
        ),
        caption=(
            "Extracted original animal "
            "on white"
        ),
        use_container_width=True,
    )


# ============================================================
# STEP 2
# ============================================================

st.markdown("---")

st.header(
    "🦵 Step 2 — Connected walking poses"
)

st.write(
    "The proximal part of every leg is deliberately "
    "kept attached to the body. The knee has an overlap "
    "between the upper and lower pieces, and body/legs "
    "receive the same bob movement."
)


if st.button(
    "🦒 Generate 4 walking poses",
    type="primary",
    use_container_width=True,
):

    with st.spinner(
        "Building connected giraffe walking poses..."
    ):

        # White is NOT used here because these are
        # diagnostic walking poses.
        #
        # Use the cleaned original background.
        clean_background = (
            remove_animal_from_background(
                image,
                animal,
                animal_bbox,
            )
        )

        poses = []

        for t in (
            0.00,
            0.25,
            0.50,
            0.75,
        ):

            pose = create_walking_pose(
                clean_background,
                body_sprite,
                animal_bbox,
                prepared_legs,
                t,
            )

            poses.append(
                pose
            )

        st.session_state[
            "walk_keyframes"
        ] = poses

        st.session_state.pop(
            "frames",
            None,
        )


if "walk_keyframes" in st.session_state:

    cols = st.columns(4)

    for i, frame in enumerate(
        st.session_state[
            "walk_keyframes"
        ]
    ):

        with cols[i]:

            st.image(
                cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB,
                ),
                caption=f"Pose {i + 1}",
                use_container_width=True,
            )

    st.download_button(
        "⬇️ Download 4 walking poses",
        zip_frames(
            st.session_state[
                "walk_keyframes"
            ],
            "connected_walk_pose",
        ),
        "connected_giraffe_walk_poses.zip",
        "application/zip",
        use_container_width=True,
    )


# ============================================================
# STEP 3
# ============================================================

st.markdown("---")

st.header(
    "🎞️ Step 3 — Full animation"
)

st.write(
    """
The final animation is designed to behave like this:

**White → giraffe walks in → reaches exact original location
→ walks/stands → settles → original image appears.**
"""
)


if st.button(
    "🚀 Render full animation",
    type="primary",
    use_container_width=True,
):

    with st.spinner(
        "Rendering complete animation..."
    ):

        frames = build_animation(
            image=image,
            animal=animal,
            animal_bbox=animal_bbox,
            body_sprite=body_sprite,
            prepared_legs=prepared_legs,
            total_frames=TOTAL_FRAMES,
            walk_cycles=WALK_CYCLES,
            mode=ANIMATION_MODE,
            walk_in_fraction=WALK_IN_FRACTION,
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

    st.markdown("---")

    st.header(
        "🎉 Final Animation"
    )

    st.image(
        gif,
        caption=(
            "White canvas → walking animal → "
            "exact original position → scenery merge"
        ),
        use_container_width=True,
    )

    a, b, c = st.columns(3)

    with a:

        st.download_button(
            "⬇️ Download GIF",
            gif,
            "hand_drawn_giraffe_walk_v4.gif",
            "image/gif",
            use_container_width=True,
        )

    with b:

        mp4 = mp4_bytes(
            frames,
            FPS,
        )

        if mp4:

            st.download_button(
                "⬇️ Download MP4",
                mp4,
                "hand_drawn_giraffe_walk_v4.mp4",
                "video/mp4",
                use_container_width=True,
            )

        else:

            st.info(
                "MP4 unavailable in this environment."
            )

    with c:

        st.download_button(
            "⬇️ Download PNG frames",
            zip_frames(
                frames,
                "giraffe_walk_frame",
            ),
            "hand_drawn_giraffe_walk_v4_frames.zip",
            "application/zip",
            use_container_width=True,
        )

    # ========================================================
    # FRAME PREVIEW
    # ========================================================

    st.markdown(
        "### 🖼️ Animation frame preview"
    )

    indices = np.linspace(
        0,
        len(frames) - 1,
        min(
            16,
            len(frames),
        ),
        dtype=int,
    )

    cols = st.columns(4)

    for n, index in enumerate(
        indices
    ):

        with cols[
            n % 4
        ]:

            st.image(
                cv2.cvtColor(
                    frames[index],
                    cv2.COLOR_BGR2RGB,
                ),
                caption=f"Frame {index + 1}",
                use_container_width=True,
            )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "v4 — Connected original-pixel animal animation. "
    "Gemini provides anatomy/geometry only. "
    "Python performs all movement and compositing."
)

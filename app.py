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

st.title(
    "🦒 Hand-Drawn Animal Walk Animator — Solid Connected Walk v8"
)

st.caption(
    "Gemini identifies animal anatomy. Python animates the ORIGINAL drawing "
    "using solid connected limb pivots. The original animal is hidden during "
    "walking to prevent duplicate-leg ghosting."
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

st.sidebar.markdown("### 🦵 Leg Movement & Gait")

STEP_ANGLE = st.sidebar.slider(
    "Leg swing angle",
    2.0,
    25.0,
    10.0,
    0.5,
)

BODY_BOB = st.sidebar.slider(
    "Body vertical bob",
    0.0,
    0.04,
    0.006,
    0.001,
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
    actions = getattr(
        obj,
        "supported_actions",
        None,
    )

    if actions is None:
        actions = getattr(
            obj,
            "supportedActions",
            None,
        )

    try:
        return [
            str(x)
            for x in (actions or [])
        ]
    except Exception:
        return []


def discover_models(
    client: genai.Client,
) -> List[str]:

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


def model_score(
    name: str,
) -> Tuple[int, str]:

    x = (
        name
        .lower()
        .replace("models/", "")
    )

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


def safe_text(
    response: Any,
) -> str:

    text = getattr(
        response,
        "text",
        None,
    )

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
                    parts.append(
                        str(value)
                    )

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
Analyze this SINGLE hand-drawn animal scene for a 2D cut-out animation system.

Identify the animal, its body, and its visible legs.

Return geometry only.

Coordinates must be normalized 0..100 and represented as [y,x].

IMPORTANT:

1. Identify the main visible animal.

2. The animal polygon must surround the animal completely,
   including torso, head, neck, tail and visible limbs,
   but excluding background and floor shadows.

3. Identify EVERY clearly visible leg.

4. Each leg must have a tight polygon around the actual visible leg
   without floor shadows.

5. Each leg must specify:

   - proximal:
     exact point where the leg attaches to the body.

   - middle:
     knee/bend location.

   - distal:
     hoof/foot location.

6. Provide side tags:

   front_left
   front_right
   back_left
   back_right

7. The BODY polygon should describe the torso/body region,
   NOT the complete legs.

8. Do not include floor shadows as animal parts.

9. Do not invent hidden legs that cannot be seen.

10. Preserve the exact original position of the animal.

Return ONLY valid JSON with this structure:

{
  "identified_character": "giraffe",

  "locomotion_profile": {
    "type": "quadruped",
    "stride_multiplier": 1.0
  },

  "animal_bbox": [
    ymin,
    xmin,
    ymax,
    xmax
  ],

  "animal_polygon": [
    [y,x]
  ],

  "parts": [

    {
      "name": "body",
      "type": "body",
      "polygon": [
        [y,x]
      ]
    },

    {
      "name": "front_left_leg",
      "type": "leg",
      "side": "front_left",
      "polygon": [
        [y,x]
      ],
      "joints": {
        "proximal": [y,x],
        "middle": [y,x],
        "distal": [y,x]
      }
    }

  ],

  "notes": "..."
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

            text = safe_text(
                response
            )

            data = json.loads(
                clean_json_text(text)
            )

            if not isinstance(
                data,
                dict,
            ):
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

    with st.expander(
        "Model attempts"
    ):

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
            isinstance(
                p,
                (list, tuple),
            )
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
        max(
            0,
            x - margin,
        ),
        max(
            0,
            y - margin,
        ),
        min(
            width,
            x + bw + margin,
        ),
        min(
            height,
            y + bh + margin,
        ),
    )


def bbox_norm(
    bbox,
    width,
    height,
    margin=10,
):

    if (
        not isinstance(
            bbox,
            list,
        )
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
        max(
            0,
            x1 - margin,
        ),
        max(
            0,
            y1 - margin,
        ),
        min(
            width,
            x2 + margin,
        ),
        min(
            height,
            y2 + margin,
        ),
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

    _, S, V = cv2.split(hsv)

    colored = (
        S > 35
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

    dark = (
        V < 145
    ).astype(
        np.uint8
    ) * 255

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

        source = np.dstack(
            [
                source,
                np.full(
                    source.shape[:2],
                    255,
                    dtype=np.uint8,
                ),
            ]
        )

    if source.shape[2] < 4:
        return destination

    H, W = destination.shape[:2]
    sh, sw = source.shape[:2]

    x = int(x)
    y = int(y)

    x1 = max(
        0,
        x,
    )

    y1 = max(
        0,
        y,
    )

    x2 = min(
        W,
        x + sw,
    )

    y2 = min(
        H,
        y + sh,
    )

    if x1 >= x2 or y1 >= y2:
        return destination

    sx1 = x1 - x
    sy1 = y1 - y

    sx2 = sx1 + (
        x2 - x1
    )

    sy2 = sy1 + (
        y2 - y1
    )

    src = source[
        sy1:sy2,
        sx1:sx2,
    ]

    src_rgb = src[
        :, :, :3
    ].astype(
        np.float32
    )

    alpha = (
        src[:, :, 3].astype(
            np.float32
        )
        / 255.0
    )

    alpha = alpha[:, :, None]

    dst = destination[
        y1:y2,
        x1:x2,
    ].astype(
        np.float32
    )

    if dst.ndim != 3:
        return destination

    dst = dst[:, :, :3]

    result = (
        src_rgb * alpha
        + dst * (
            1.0 - alpha
        )
    )

    destination[
        y1:y2,
        x1:x2,
    ] = np.clip(
        result,
        0,
        255,
    ).astype(
        np.uint8
    )

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
        borderValue=(
            0,
            0,
            0,
            0,
        ),
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


# ============================================================
# LEG PREPARATION
# ============================================================

def leg_parts(
    scene,
):

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

            result.append(
                part
            )

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
        or "proximal" not in joints
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

    margin = max(
        35,
        min(h, w) // 40,
    )

    bbox = bbox_poly(
        polygon,
        w,
        h,
        margin=margin,
    )

    x1, y1, x2, y2 = bbox

    if x2 <= x1 or y2 <= y1:
        return None

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

    # Slight expansion keeps the original hand-drawn stroke
    # connected after rotation.
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
        for key in joints
        if isinstance(
            joints[key],
            (list, tuple),
        )
        and len(joints[key]) >= 2
    }

    local_joints = {
        key: (
            global_joints[key]
            - np.array(
                [
                    x1,
                    y1,
                ],
                dtype=np.float32,
            )
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
# SOLID CONNECTED PENDULUM LEG
# ============================================================

def swing_leg(
    leg,
    swing_angle,
):
    """
    ORIGINAL GAIT ALGORITHM.

    Each leg remains a single solid cut-out and rotates
    around the proximal joint.
    """

    sprite = leg["sprite"]
    joints = leg["joints"]

    if "proximal" in joints:

        pivot = (
            float(
                joints["proximal"][0]
            ),
            float(
                joints["proximal"][1]
            ),
        )

    else:

        pivot = (
            float(
                sprite.shape[1] // 2
            ),
            0.0,
        )

    matrix = rotation_matrix(
        swing_angle,
        pivot,
    )

    rotated = warp_rgba(
        sprite,
        sprite[:, :, 3].copy(),
        matrix,
    )

    return (
        rotated,
        leg["bbox"][0],
        leg["bbox"][1],
    )


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
# BODY-ONLY SPRITE
# ============================================================

def make_body_sprite(
    animal,
    animal_bbox,
    prepared_legs,
):
    """
    Critical anti-ghosting stage.

    The complete extracted animal contains the original
    stationary legs.

    We remove those leg regions from the body sprite so
    only the animated leg cut-outs can appear during walking.

    A very small region around each proximal attachment is
    retained so the animated leg remains visually connected
    to the torso.
    """

    body = animal.copy()

    if (
        body.ndim != 3
        or body.shape[2] < 4
    ):
        return body

    alpha = body[
        :, :, 3
    ].copy()

    if not prepared_legs:
        return body

    body_h, body_w = alpha.shape

    bx, by, _, _ = animal_bbox

    leg_union = np.zeros(
        (body_h, body_w),
        dtype=np.uint8,
    )

    # --------------------------------------------------------
    # Build a union of every extracted leg mask.
    # --------------------------------------------------------

    for leg in prepared_legs:

        leg_mask = leg.get(
            "mask"
        )

        if leg_mask is None:
            continue

        lx, ly = leg["bbox"][0], leg["bbox"][1]

        local_x = int(
            lx - bx
        )

        local_y = int(
            ly - by
        )

        lh, lw = leg_mask.shape[:2]

        x1 = max(
            0,
            local_x,
        )

        y1 = max(
            0,
            local_y,
        )

        x2 = min(
            body_w,
            local_x + lw,
        )

        y2 = min(
            body_h,
            local_y + lh,
        )

        if (
            x1 >= x2
            or y1 >= y2
        ):
            continue

        sx1 = x1 - local_x
        sy1 = y1 - local_y

        sx2 = sx1 + (
            x2 - x1
        )

        sy2 = sy1 + (
            y2 - y1
        )

        leg_union[
            y1:y2,
            x1:x2
        ] = np.maximum(
            leg_union[
                y1:y2,
                x1:x2
            ],
            leg_mask[
                sy1:sy2,
                sx1:sx2
            ],
        )

    # --------------------------------------------------------
    # Expand removal slightly.
    #
    # This is important because the animal extraction and
    # individual leg extraction may not have identical edges.
    # --------------------------------------------------------

    removal_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (7, 7),
    )

    leg_union = cv2.dilate(
        leg_union,
        removal_kernel,
        iterations=1,
    )

    # --------------------------------------------------------
    # Protect ONLY a small proximal attachment area.
    # --------------------------------------------------------

    protected = np.zeros_like(
        leg_union
    )

    for leg in prepared_legs:

        global_joints = leg.get(
            "global",
            {},
        )

        proximal = global_joints.get(
            "proximal"
        )

        if proximal is None:
            continue

        px = int(
            proximal[0] - bx
        )

        py = int(
            proximal[1] - by
        )

        # Small attachment radius.
        #
        # Deliberately much smaller than the old //10 rule.
        leg_h, leg_w = leg[
            "sprite"
        ].shape[:2]

        radius = int(
            np.clip(
                min(
                    leg_h,
                    leg_w,
                ) * 0.035,
                4,
                10,
            )
        )

        cv2.circle(
            protected,
            (
                px,
                py,
            ),
            radius,
            255,
            -1,
        )

    # --------------------------------------------------------
    # Actual removal.
    # --------------------------------------------------------

    removable = cv2.bitwise_and(
        leg_union,
        cv2.bitwise_not(
            protected
        ),
    )

    alpha[
        removable > 15
    ] = 0

    # Remove faint residual pixels around the cut.
    alpha = cv2.GaussianBlur(
        alpha,
        (3, 3),
        0,
    )

    # Do not allow nearly invisible remnants to become
    # ghost limbs after compositing.
    alpha[
        alpha < 18
    ] = 0

    body[
        :, :, 3
    ] = alpha

    return body


# ============================================================
# WALKING PHASE & GAIT
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


def smooth_gait(
    value,
):

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
    profile=None,
):

    frame = background.copy()

    h, w = background.shape[:2]

    stride_mult = 1.0

    if (
        profile
        and isinstance(
            profile,
            dict,
        )
    ):

        try:

            stride_mult = float(
                profile.get(
                    "stride_multiplier",
                    1.0,
                )
            )

        except Exception:

            stride_mult = 1.0

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

    # ========================================================
    # LEGS FIRST
    # ========================================================

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
            * stride_mult
        )

        if "back" in side:
            swing *= 0.90

        rotated_leg, lx, ly = (
            swing_leg(
                leg,
                swing,
            )
        )

        frame = alpha_over(
            frame,
            rotated_leg,
            lx,
            ly + bob,
        )

    # ========================================================
    # BODY LAST
    # ========================================================

    frame = alpha_over(
        frame,
        body_sprite,
        bx,
        by + bob,
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

    for leg in prepared_legs:

        rotated_leg, lx, ly = (
            swing_leg(
                leg,
                0.0,
            )
        )

        frame = alpha_over(
            frame,
            rotated_leg,
            lx,
            ly,
        )

    frame = alpha_over(
        frame,
        body_sprite,
        bx,
        by,
    )

    return frame


# ============================================================
# EASING
# ============================================================

def ease_in_out(
    t,
):

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


# ============================================================
# TRANSLATE COMPLETE WALKING ANIMAL
# ============================================================

def translate_walking_pose(
    pose,
    dx,
    dy,
):

    H, W = pose.shape[:2]

    matrix = np.float32(
        [
            [
                1.0,
                0.0,
                dx,
            ],
            [
                0.0,
                1.0,
                dy,
            ],
        ]
    )

    return cv2.warpAffine(
        pose,
        matrix,
        (W, H),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(
            255,
            255,
            255,
        ),
    )


# ============================================================
# WALK-IN FRAME
# ============================================================

def create_moving_walking_frame(
    white_canvas,
    body_sprite,
    body_bbox,
    prepared_legs,
    t,
    travel_p,
    profile=None,
):

    H, W = white_canvas.shape[:2]

    # --------------------------------------------------------
    # Build the walking animal at its EXACT ORIGINAL position.
    # --------------------------------------------------------

    pose = create_walking_pose(
        white_canvas,
        body_sprite,
        body_bbox,
        prepared_legs,
        t,
        profile=profile,
    )

    bx, by, x2, y2 = body_bbox

    animal_width = max(
        1,
        x2 - bx,
    )

    extra = int(
        W
        * ENTRY_EXTRA_DISTANCE
    )

    if ENTRY_SIDE == "Left":

        start_x = (
            -animal_width
            - extra
        )

    else:

        start_x = (
            W
            + extra
        )

    # --------------------------------------------------------
    # Smooth movement from outside the canvas to original X.
    # --------------------------------------------------------

    p = ease_in_out(
        travel_p
    )

    current_x = (
        start_x
        + (
            bx
            - start_x
        )
        * p
    )

    dx = (
        current_x
        - bx
    )

    # Tiny vertical entrance arc.
    dy = (
        math.sin(
            p
            * math.pi
        )
        * H
        * 0.004
    )

    return translate_walking_pose(
        pose,
        dx,
        dy,
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

    return cv2.addWeighted(
        animated_frame,
        1.0 - p,
        original,
        p,
        0,
    )


# ============================================================
# FRAME ALLOCATION
# ============================================================

def allocate_frames(
    total_frames,
    mode,
    walk_fraction,
    merge_fraction,
):
    """
    Allocate EXACTLY total_frames.

    Unlike the previous version, no phase is allowed to push
    the total over TOTAL_FRAMES and then get blindly truncated.
    """

    if mode == "Walk in place only":

        return (
            0,
            total_frames,
            0,
        )

    requested_walk = max(
        6,
        int(
            round(
                total_frames
                * walk_fraction
            )
        ),
    )

    requested_merge = 0

    if mode == (
        "White canvas → walk in → stand → merge"
    ):

        requested_merge = max(
            5,
            int(
                round(
                    total_frames
                    * merge_fraction
                )
            ),
        )

    # Keep at least one frame for the middle phase.
    maximum_walk = max(
        1,
        total_frames
        - requested_merge
        - 1,
    )

    walk_n = min(
        requested_walk,
        maximum_walk,
    )

    maximum_merge = max(
        0,
        total_frames
        - walk_n
        - 1,
    )

    merge_n = min(
        requested_merge,
        maximum_merge,
    )

    stand_n = (
        total_frames
        - walk_n
        - merge_n
    )

    # Final safety.
    if stand_n < 1:

        deficit = (
            1
            - stand_n
        )

        reduce_merge = min(
            deficit,
            max(
                0,
                merge_n - 1,
            ),
        )

        merge_n -= reduce_merge
        deficit -= reduce_merge

        if deficit > 0:

            walk_n = max(
                1,
                walk_n - deficit,
            )

    stand_n = (
        total_frames
        - walk_n
        - merge_n
    )

    return (
        walk_n,
        stand_n,
        merge_n,
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
    locomotion_profile=None,
):

    H, W = image.shape[:2]

    # ========================================================
    # PURE WHITE CANVAS
    # ========================================================

    white = np.full(
        (
            H,
            W,
            3,
        ),
        255,
        dtype=np.uint8,
    )

    # ========================================================
    # EXACT FRAME ALLOCATION
    # ========================================================

    walk_n, stand_n, merge_n = (
        allocate_frames(
            total_frames,
            mode,
            walk_in_fraction,
            merge_fraction,
        )
    )

    frames = []

    # ========================================================
    # PHASE 0
    #
    # GUARANTEED PURE WHITE INTRO
    #
    # This makes the animation visibly begin from a blank
    # canvas rather than immediately showing an off-screen
    # translated object.
    # ========================================================

    if walk_n > 0:

        intro_n = min(
            3,
            max(
                1,
                walk_n // 5,
            ),
        )

        for _ in range(
            intro_n
        ):

            frames.append(
                white.copy()
            )

    else:

        intro_n = 0

    # Remaining walk-in frames.
    actual_walk_frames = (
        walk_n
        - intro_n
    )

    # ========================================================
    # PHASE 1
    #
    # ANIMAL WALKS IN FROM OUTSIDE
    # ========================================================

    for i in range(
        actual_walk_frames
    ):

        if actual_walk_frames <= 1:

            travel_p = 1.0

        else:

            travel_p = (
                i
                / (
                    actual_walk_frames
                    - 1
                )
            )

        # At least one complete gait cycle occurs during entry.
        gait_cycles = max(
            1.0,
            float(
                walk_cycles
            ),
        )

        gait_t = (
            travel_p
            * gait_cycles
        )

        frame = (
            create_moving_walking_frame(
                white_canvas=white,
                body_sprite=body_sprite,
                body_bbox=animal_bbox,
                prepared_legs=prepared_legs,
                t=gait_t,
                travel_p=travel_p,
                profile=locomotion_profile,
            )
        )

        frames.append(
            frame
        )

    # ========================================================
    # PHASE 2
    #
    # ANIMAL IS NOW AT EXACT ORIGINAL POSITION
    #
    # WHITE BACKGROUND REMAINS.
    # ========================================================

    if stand_n > 0:

        if walk_cycles <= 0:

            for _ in range(
                stand_n
            ):

                frame = (
                    create_standing_pose(
                        white,
                        body_sprite,
                        animal_bbox,
                        prepared_legs,
                    )
                )

                frames.append(
                    frame
                )

        else:

            for i in range(
                stand_n
            ):

                t = (
                    i
                    / max(
                        1,
                        stand_n,
                    )
                    * walk_cycles
                )

                frame = (
                    create_walking_pose(
                        white,
                        body_sprite,
                        animal_bbox,
                        prepared_legs,
                        t,
                        profile=locomotion_profile,
                    )
                )

                frames.append(
                    frame
                )

    # ========================================================
    # PHASE 3
    #
    # FINAL EXACT-POSITION ISOLATED FRAME
    # ========================================================

    if frames:

        final_isolated_frame = (
            frames[-1].copy()
        )

    else:

        final_isolated_frame = (
            create_standing_pose(
                white,
                body_sprite,
                animal_bbox,
                prepared_legs,
            )
        )

    # ========================================================
    # PHASE 4
    #
    # COMPLETE ORIGINAL SCENERY MERGE
    # ========================================================

    if merge_n > 0:

        for i in range(
            merge_n
        ):

            if merge_n <= 1:

                p = 1.0

            else:

                p = (
                    i
                    / (
                        merge_n
                        - 1
                    )
                )

            frame = smooth_merge(
                final_isolated_frame,
                image,
                p,
            )

            frames.append(
                frame
            )

    # ========================================================
    # EXACT FRAME COUNT
    # ========================================================

    if len(frames) > total_frames:

        frames = frames[
            :total_frames
        ]

    while len(frames) < total_frames:

        if merge_n > 0:

            frames.append(
                image.copy()
            )

        elif frames:

            frames.append(
                frames[-1].copy()
            )

        else:

            frames.append(
                white.copy()
            )

    # ========================================================
    # GUARANTEE FINAL FRAME
    # ========================================================

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
        (
            w,
            h,
        ),
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
            (
                0,
                180,
                0,
            ),
            max(
                2,
                min(
                    h,
                    w,
                )
                // 300,
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
                (
                    255,
                    120,
                    0,
                ),
                max(
                    1,
                    min(
                        h,
                        w,
                    )
                    // 450,
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

                point = pt_px(
                    joint,
                    w,
                    h,
                ).astype(
                    int
                )

                x = int(
                    point[0]
                )

                y = int(
                    point[1]
                )

                cv2.circle(
                    output,
                    (
                        x,
                        y,
                    ),
                    6,
                    (
                        0,
                        0,
                        255,
                    ),
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
                    (
                        0,
                        0,
                        255,
                    ),
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
        "Upload the drawing containing your animal "
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
    ).convert(
        "RGB"
    )

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
### 🎬 Solid Connected Animation

**1. Completely white canvas**

⬇️

**2. Animal enters while already walking**

⬇️

**3. Animal reaches exact original position**

⬇️

**4. Legs continue swinging as solid units**

⬇️

**5. Complete original scenery fades in**

⬇️

**6. Final frame = exact original drawing**
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
        "Gemini is analyzing animal anatomy..."
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
        "Orange = leg polygons | "
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
        "Gemini did not return usable leg polygons and joints."
    )

    st.stop()


st.success(
    f"Prepared {len(prepared_legs)} solid connected leg cut-outs."
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
# STEP 2
# ============================================================

st.markdown("---")

st.header(
    "🦵 Step 2 — Solid connected walking poses"
)

st.write(
    "The preview uses the same pure-white isolated composition "
    "as the final animation. Original stationary legs are removed "
    "from the body sprite before the animated legs are rendered."
)


if st.button(
    "🦒 Generate 4 connected walking poses",
    type="primary",
    use_container_width=True,
):

    with st.spinner(
        "Building solid walking poses..."
    ):

        white_preview = np.full(
            image.shape,
            255,
            dtype=np.uint8,
        )

        poses = []

        locomotion_profile = (
            scene.get(
                "locomotion_profile",
                {},
            )
        )

        for t in (
            0.00,
            0.25,
            0.50,
            0.75,
        ):

            pose = create_walking_pose(
                white_preview,
                body_sprite,
                animal_bbox,
                prepared_legs,
                t,
                profile=locomotion_profile,
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
            "connected_pose",
        ),
        "connected_animal_walk_poses.zip",
        "application/zip",
        use_container_width=True,
    )


# ============================================================
# STEP 3
# ============================================================

st.markdown("---")

st.header(
    "🎞️ Step 3 — Full animation render"
)


if st.button(
    "🚀 Render full animation",
    type="primary",
    use_container_width=True,
):

    with st.spinner(
        "Rendering complete animation..."
    ):

        locomotion_profile = (
            scene.get(
                "locomotion_profile",
                {},
            )
        )

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
            locomotion_profile=locomotion_profile,
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
            "Pure white → animal walks in while animated → "
            "exact original position → scenery gradually appears → "
            "original drawing"
        ),
        use_container_width=True,
    )

    a, b, c = st.columns(3)

    with a:

        st.download_button(
            "⬇️ Download GIF",
            gif,
            "connected_animal_walk.gif",
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
                "connected_animal_walk.mp4",
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
                "animal_walk_frame",
            ),
            "connected_animal_walk_frames.zip",
            "application/zip",
            use_container_width=True,
        )

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
    "v8 — Solid connected animation. Gemini provides anatomy and "
    "joints. The original pendulum leg algorithm is preserved. "
    "Stationary leg pixels are removed from the body sprite before "
    "walking, the isolated animal enters over pure white, reaches "
    "its original position, and the complete original scenery is "
    "restored only during the final merge."
)

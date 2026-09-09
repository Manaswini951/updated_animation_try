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

st.title("🦒 Hand-Drawn Animal Walk Animator — Solid Connected Walk v8")

st.caption(
    "Gemini identifies animal anatomy. Python animates the ORIGINAL drawing "
    "using solid connected limb pivots. The original animal is hidden during "
    "walking to prevent duplicate-leg ghosting."
)


# ============================================================
# SIDEBAR SETTINGS
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
    "Total animation frames",
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
    "Walking cycles",
    0,
    5,
    2,
)

STEP_ANGLE = st.sidebar.slider(
    "Leg swing angle",
    2,
    25,
    10,
)

BODY_BOB = st.sidebar.slider(
    "Body bob",
    0.0,
    0.04,
    0.006,
    0.001,
)

WALK_IN_FRACTION = st.sidebar.slider(
    "Walk-in fraction",
    0.10,
    0.50,
    0.25,
    0.05,
)

STAND_FRACTION = st.sidebar.slider(
    "Stand fraction",
    0.10,
    0.60,
    0.45,
    0.05,
)

MERGE_FRACTION = st.sidebar.slider(
    "Merge fraction",
    0.05,
    0.35,
    0.18,
    0.05,
)

INK_DILATION = st.sidebar.slider(
    "Ink dilation",
    1,
    7,
    3,
)

SHADOW_SUPPRESSION = st.sidebar.checkbox(
    "Suppress floor/background shadows",
    value=True,
)

ENTRY_SIDE = st.sidebar.selectbox(
    "Animal enters from",
    ["Left", "Right"],
)

ENTRY_EXTRA_DISTANCE = st.sidebar.slider(
    "Extra entry distance",
    0.00,
    0.30,
    0.08,
    0.01,
)


# ============================================================
# SESSION STATE
# ============================================================

if "scene" not in st.session_state:
    st.session_state.scene = None

if "walk_keyframes" not in st.session_state:
    st.session_state.walk_keyframes = None

if "frames" not in st.session_state:
    st.session_state.frames = None


# ============================================================
# GEMINI HELPERS
# ============================================================

def clean_json_text(text: str) -> str:
    text = (text or "").strip()

    if text.startswith("```"):
        lines = text.splitlines()

        if lines and lines[0].startswith("```"):
            lines = lines[1:]

        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        text = "\n".join(lines).strip()

    return text


def model_name(obj: Any) -> str:
    name = getattr(obj, "name", "") or ""
    return str(name).strip()


def model_actions(obj: Any) -> List[str]:
    actions = getattr(obj, "supported_actions", None)

    if actions is None:
        actions = getattr(obj, "supportedActions", None)

    if actions is None:
        return []

    try:
        return [str(a) for a in actions]
    except Exception:
        return []


def discover_models(client: genai.Client) -> List[str]:
    discovered = []

    try:
        for model_obj in client.models.list():
            name = model_name(model_obj)

            if not name:
                continue

            actions = model_actions(model_obj)

            if actions and "generateContent" not in actions:
                continue

            discovered.append(name)

    except Exception:
        return []

    unique = []
    seen = set()

    for name in discovered:
        key = name.lower()

        if key not in seen:
            seen.add(key)
            unique.append(name)

    return unique


def model_score(name: str) -> Tuple[int, str]:
    n = name.lower().replace("models/", "")

    if "gemini-3.7-flash" in n:
        return 0, n

    if "gemini-3.6-flash" in n:
        return 1, n

    if "gemini-3.5-flash" in n:
        return 2, n

    if "gemini-3.1-flash" in n and "lite" not in n:
        return 3, n

    if "gemini-3.1-flash-lite" in n:
        return 4, n

    if "gemini-3" in n and "flash" in n:
        return 5, n

    if "gemini-2.5-flash" in n and "lite" not in n:
        return 6, n

    if "gemini-2.5-flash-lite" in n:
        return 7, n

    if "gemini-2.5" in n:
        return 8, n

    if "gemini-2" in n and "flash" in n:
        return 10, n

    if "gemini" in n and "pro" in n:
        return 15, n

    if "gemini" in n and "flash" in n:
        return 20, n

    if "gemini" in n:
        return 30, n

    return 100, n


def safe_text(response: Any) -> str:
    text = getattr(response, "text", None)

    if text:
        return str(text)

    try:
        pieces = []

        for candidate in getattr(response, "candidates", []) or []:
            content = getattr(candidate, "content", None)

            for part in getattr(content, "parts", []) or []:
                part_text = getattr(part, "text", None)

                if part_text:
                    pieces.append(str(part_text))

        return "\n".join(pieces).strip()

    except Exception:
        return ""


# ============================================================
# GEMINI SCENE ANALYSIS
# ============================================================

def analyze_scene(
    image_bytes: bytes,
    api_key: str,
) -> Optional[Dict[str, Any]]:

    if not api_key:
        st.error("Please enter your Gemini API key.")
        return None

    try:
        client = genai.Client(api_key=api_key)
    except Exception as exc:
        st.error(f"Could not initialize Gemini: {exc}")
        return None

    discovered = discover_models(client)

    if not discovered:
        st.error(
            "Gemini API did not return any models supporting generateContent. "
            "Check the API key and Gemini API access."
        )
        return None

    discovered.sort(key=model_score)

    prompt = r"""
You are analyzing a SINGLE hand-drawn animal scene.

The goal is NOT to redraw the animal.

The goal is to provide accurate geometry so Python can cut pieces from the
ORIGINAL drawing and animate those original pixels.

IMPORTANT:

- Preserve the artist's original drawing.
- Do NOT invent missing anatomy.
- Do NOT create new artwork.
- Estimate geometry from the visible pixels.
- Identify the main animal/character.
- Identify the complete visible animal.
- Identify every visible leg separately.
- Do NOT invent hidden legs.
- Do NOT include floor shadows as animal anatomy.
- If a leg is visible, identify three joints:
  proximal, middle, distal.
- For quadrupeds these correspond approximately to hip/shoulder,
  knee/elbow, and ankle/wrist/hoof.
- Coordinates must be normalized 0..100 as [y,x].
- Polygons should follow the visible drawing tightly.
- The animal polygon should surround the whole visible animal.
- Include enough margin to avoid cutting hand-drawn strokes.
- If tail/head/neck are clearly visible, they may be identified.
- Do not make the animal bbox equal to the entire image unless appropriate.

Return ONLY valid JSON.

Required structure:

{
  "identified_character": "giraffe",
  "locomotion_profile": {
    "type": "quadruped",
    "stride_multiplier": 1.0
  },
  "animal_bbox": [ymin, xmin, ymax, xmax],
  "animal_polygon": [[y,x], [y,x], ...],
  "parts": [
    {
      "name": "body",
      "type": "body",
      "polygon": [[y,x], ...]
    },
    {
      "name": "front_left_leg",
      "type": "leg",
      "side": "front_left",
      "polygon": [[y,x], ...],
      "joints": {
        "proximal": [y,x],
        "middle": [y,x],
        "distal": [y,x]
      }
    }
  ],
  "notes": "short description"
}

For four-legged animals use these names whenever possible:

front_left_leg
front_right_leg
back_left_leg
back_right_leg

If only two or three legs are visible, return only those visible legs.

Do NOT invent hidden legs.

If no animal is visible:

{
  "identified_character": "none",
  "animal_bbox": null,
  "animal_polygon": [],
  "parts": [],
  "notes": "No clear animal detected"
}
"""

    errors = []

    for current_model in discovered:

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

            text = clean_json_text(
                safe_text(response)
            )

            if not text:
                raise ValueError(
                    "Gemini returned an empty response."
                )

            data = json.loads(text)

            if not isinstance(data, dict):
                raise ValueError(
                    "Gemini response was not a JSON object."
                )

            st.success(
                f"✅ Gemini analysis completed with `{current_model}`"
            )

            return data

        except Exception as exc:

            errors.append(
                f"{current_model}: "
                f"{str(exc).replace(chr(10), ' ')}"
            )

            continue

    st.error(
        "Gemini analysis failed after trying all available models."
    )

    with st.expander("Show model attempts"):
        for error in errors:
            st.code(error)

    return None


# ============================================================
# GEOMETRY HELPERS
# ============================================================

def pt_px(
    p: List[float],
    width: int,
    height: int,
) -> Tuple[int, int]:

    y = float(np.clip(p[0], 0, 100))
    x = float(np.clip(p[1], 0, 100))

    return (
        int(x * width / 100.0),
        int(y * height / 100.0),
    )


def poly_px(
    polygon: List[List[float]],
    width: int,
    height: int,
) -> np.ndarray:

    pts = []

    for p in polygon:

        if isinstance(p, (list, tuple)) and len(p) >= 2:
            x, y = pt_px(p, width, height)
            pts.append([x, y])

    if len(pts) < 3:
        return np.empty((0, 2), dtype=np.int32)

    return np.asarray(pts, dtype=np.int32)


def poly_mask(
    shape: Tuple[int, int],
    polygon: List[List[float]],
    dilation: int = 0,
) -> np.ndarray:

    h, w = shape[:2]

    mask = np.zeros(
        (h, w),
        dtype=np.uint8,
    )

    pts = poly_px(
        polygon,
        w,
        h,
    )

    if len(pts) >= 3:

        cv2.fillPoly(
            mask,
            [pts],
            255,
        )

    if dilation > 0:

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (
                dilation * 2 + 1,
                dilation * 2 + 1,
            ),
        )

        mask = cv2.dilate(
            mask,
            kernel,
        )

    return mask


def bbox_poly(
    polygon: List[List[float]],
    width: int,
    height: int,
    margin: int = 10,
) -> Tuple[int, int, int, int]:

    pts = poly_px(
        polygon,
        width,
        height,
    )

    if len(pts) == 0:
        return 0, 0, width, height

    x, y, bw, bh = cv2.boundingRect(pts)

    x1 = max(0, x - margin)
    y1 = max(0, y - margin)

    x2 = min(
        width,
        x + bw + margin,
    )

    y2 = min(
        height,
        y + bh + margin,
    )

    return x1, y1, x2, y2


def bbox_norm(
    bbox: Optional[List[float]],
    width: int,
    height: int,
    margin: int = 10,
) -> Tuple[int, int, int, int]:

    if not bbox or len(bbox) != 4:
        return 0, 0, width, height

    ymin, xmin, ymax, xmax = [
        float(v) for v in bbox
    ]

    x1 = int(xmin * width / 100)
    y1 = int(ymin * height / 100)

    x2 = int(xmax * width / 100)
    y2 = int(ymax * height / 100)

    x1 = max(
        0,
        x1 - margin,
    )

    y1 = max(
        0,
        y1 - margin,
    )

    x2 = min(
        width,
        x2 + margin,
    )

    y2 = min(
        height,
        y2 + margin,
    )

    return x1, y1, x2, y2


def rotation_matrix(
    angle: float,
    center: Tuple[float, float],
) -> np.ndarray:

    return cv2.getRotationMatrix2D(
        center,
        angle,
        1.0,
    )


# ============================================================
# FOREGROUND / INK EXTRACTION
# ============================================================

def ink_mask(
    image_bgr: np.ndarray,
    region_mask: np.ndarray,
    shadow_suppress: bool = True,
    dilation: int = 3,
) -> np.ndarray:

    hsv = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2HSV,
    )

    H, S, V = cv2.split(hsv)

    colored = S > 35

    strong_color = (
        (S > 55)
        & (V > 45)
    )

    color_support = cv2.dilate(
        strong_color.astype(np.uint8),
        np.ones((5, 5), np.uint8),
    ) > 0

    dark = V < 145

    if shadow_suppress:

        dark_attached = (
            dark
            & color_support
        )

        foreground = (
            colored
            | dark_attached
        )

    else:

        foreground = (
            colored
            | dark
        )

    foreground = (
        foreground.astype(np.uint8)
        * 255
    )

    foreground = cv2.bitwise_and(
        foreground,
        region_mask,
    )

    foreground = cv2.morphologyEx(
        foreground,
        cv2.MORPH_CLOSE,
        np.ones((3, 3), np.uint8),
    )

    if dilation > 0:

        foreground = cv2.dilate(
            foreground,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (
                    dilation * 2 + 1,
                    dilation * 2 + 1,
                ),
            ),
        )

    return foreground


# ============================================================
# ALPHA COMPOSITING
# ============================================================

def alpha_over(
    destination: np.ndarray,
    source: np.ndarray,
    x: int,
    y: int,
) -> np.ndarray:

    if source is None or source.size == 0:
        return destination

    sh, sw = source.shape[:2]

    H, W = destination.shape[:2]

    x2 = x + sw
    y2 = y + sh

    if (
        x2 <= 0
        or y2 <= 0
        or x >= W
        or y >= H
    ):
        return destination

    cx1 = max(0, x)
    cy1 = max(0, y)

    cx2 = min(W, x2)
    cy2 = min(H, y2)

    sx1 = cx1 - x
    sy1 = cy1 - y

    sx2 = sx1 + (cx2 - cx1)
    sy2 = sy1 + (cy2 - cy1)

    src = source[
        sy1:sy2,
        sx1:sx2,
    ]

    alpha = (
        src[:, :, 3].astype(np.float32)
        / 255.0
    )[:, :, None]

    src_rgb = src[:, :, :3].astype(
        np.float32
    )

    dst = destination[
        cy1:cy2,
        cx1:cx2
    ].astype(np.float32)

    result = (
        src_rgb * alpha
        + dst * (1.0 - alpha)
    )

    destination[
        cy1:cy2,
        cx1:cx2
    ] = np.clip(
        result,
        0,
        255,
    ).astype(np.uint8)

    return destination


# ============================================================
# RGBA TRANSFORMATION
# ============================================================

def warp_rgba(
    sprite: np.ndarray,
    alpha_mask: np.ndarray,
    matrix: np.ndarray,
) -> np.ndarray:

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


# ============================================================
# LEG HELPERS
# ============================================================

def leg_parts(
    scene: Dict[str, Any],
) -> List[Dict[str, Any]]:

    result = []

    for part in scene.get("parts", []):

        if not isinstance(part, dict):
            continue

        ptype = str(
            part.get("type", "")
        ).lower()

        name = str(
            part.get("name", "")
        ).lower()

        if (
            ptype == "leg"
            or "leg" in name
        ):
            result.append(part)

    return result


def prepare_leg(
    image_bgr: np.ndarray,
    part: Dict[str, Any],
) -> Optional[Dict[str, Any]]:

    polygon = part.get(
        "polygon",
        [],
    )

    joints = part.get(
        "joints",
        {},
    )

    if len(polygon) < 3:
        return None

    if not all(
        key in joints
        for key in (
            "proximal",
            "middle",
            "distal",
        )
    ):
        return None

    h, w = image_bgr.shape[:2]

    region_mask = poly_mask(
        (h, w),
        polygon,
        dilation=max(
            2,
            INK_DILATION,
        ),
    )

    mask = ink_mask(
        image_bgr,
        region_mask,
        shadow_suppress=SHADOW_SUPPRESSION,
        dilation=INK_DILATION,
    )

    margin = max(
        35,
        min(h, w) // 40,
    )

    x1, y1, x2, y2 = bbox_poly(
        polygon,
        w,
        h,
        margin=margin,
    )

    crop = image_bgr[
        y1:y2,
        x1:x2
    ].copy()

    local_mask = mask[
        y1:y2,
        x1:x2
    ].copy()

    if crop.size == 0:
        return None

    sprite = cv2.cvtColor(
        crop,
        cv2.COLOR_BGR2BGRA,
    )

    local_alpha = cv2.GaussianBlur(
        local_mask,
        (3, 3),
        0,
    )

    sprite[:, :, 3] = local_alpha

    global_joints = {}

    local_joints = {}

    for name in (
        "proximal",
        "middle",
        "distal",
    ):

        gx, gy = pt_px(
            joints[name],
            w,
            h,
        )

        global_joints[name] = (
            gx,
            gy,
        )

        local_joints[name] = (
            gx - x1,
            gy - y1,
        )

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
        "bbox": (
            x1,
            y1,
            x2,
            y2,
        ),
        "joints": local_joints,
        "global": global_joints,
        "mask": local_alpha,
    }


# ============================================================
# BODY SPRITE
# ============================================================

def make_body_sprite(
    animal: np.ndarray,
    animal_bbox: Tuple[int, int, int, int],
    prepared_legs: List[Dict[str, Any]],
) -> np.ndarray:

    body = animal.copy()

    if body.size == 0:
        return body

    H, W = body.shape[:2]

    leg_union = np.zeros(
        (H, W),
        dtype=np.uint8,
    )

    ax1, ay1, ax2, ay2 = animal_bbox

    for leg in prepared_legs:

        x1, y1, x2, y2 = leg["bbox"]

        lx1 = max(
            0,
            x1 - ax1,
        )

        ly1 = max(
            0,
            y1 - ay1,
        )

        lx2 = min(
            W,
            x2 - ax1,
        )

        ly2 = min(
            H,
            y2 - ay1,
        )

        if lx2 <= lx1 or ly2 <= ly1:
            continue

        mask = leg["mask"]

        mh, mw = mask.shape[:2]

        tx2 = min(
            lx2,
            lx1 + mw,
        )

        ty2 = min(
            ly2,
            ly1 + mh,
        )

        if tx2 <= lx1 or ty2 <= ly1:
            continue

        leg_union[
            ly1:ty2,
            lx1:tx2
        ] = np.maximum(
            leg_union[
                ly1:ty2,
                lx1:tx2
            ],
            mask[
                :ty2 - ly1,
                :tx2 - lx1
            ],
        )

    leg_union = cv2.dilate(
        leg_union,
        np.ones((7, 7), np.uint8),
    )

    removable = leg_union.copy()

    # Protect only a small proximal attachment area.
    for leg in prepared_legs:

        proximal = leg["joints"].get(
            "proximal"
        )

        if proximal is None:
            continue

        px, py = proximal

        radius = 0.035 * min(
            leg["sprite"].shape[:2]
        )

        radius = int(
            np.clip(
                radius,
                4,
                10,
            )
        )

        cv2.circle(
            removable,
            (
                int(
                    px
                    + leg["bbox"][0]
                    - ax1
                ),
                int(
                    py
                    + leg["bbox"][1]
                    - ay1
                ),
            ),
            radius,
            0,
            -1,
        )

    body_alpha = body[
        :,
        :,
        3
    ]

    body_alpha[
        removable > 0
    ] = 0

    body_alpha = cv2.GaussianBlur(
        body_alpha,
        (3, 3),
        0,
    )

    body_alpha[
        body_alpha < 18
    ] = 0

    body[:, :, 3] = body_alpha

    return body


# ============================================================
# LEG SWING
# ============================================================

def phase_for(
    side: str,
    index: int,
) -> float:

    side = str(
        side
    ).lower()

    if "front_left" in side:
        return 0.0

    if "back_right" in side:
        return 0.0

    if "front_right" in side:
        return math.pi

    if "back_left" in side:
        return math.pi

    return (
        0.0
        if index % 2 == 0
        else math.pi
    )


def smooth_gait(
    value: float,
) -> float:

    return value * (
        0.72
        + 0.28 * abs(value)
    )


def swing_leg(
    leg: Dict[str, Any],
    swing_angle: float,
) -> Tuple[np.ndarray, int, int]:

    sprite = leg["sprite"]

    proximal = leg["joints"].get(
        "proximal"
    )

    if proximal is None:
        proximal = (
            sprite.shape[1] / 2.0,
            0.0,
        )

    matrix = rotation_matrix(
        swing_angle,
        proximal,
    )

    transformed = warp_rgba(
        sprite,
        sprite[:, :, 3],
        matrix,
    )

    px, py = proximal

    new_px = (
        matrix[0, 0] * px
        + matrix[0, 1] * py
        + matrix[0, 2]
    )

    new_py = (
        matrix[1, 0] * px
        + matrix[1, 1] * py
        + matrix[1, 2]
    )

    bx, by, _, _ = leg["bbox"]

    target_x = int(
        round(
            bx + px - new_px
        )
    )

    target_y = int(
        round(
            by + py - new_py
        )
    )

    return (
        transformed,
        target_x,
        target_y,
    )


# ============================================================
# WALKING POSE
# ============================================================

def create_walking_pose(
    background: np.ndarray,
    body_sprite: np.ndarray,
    body_bbox: Tuple[int, int, int, int],
    prepared_legs: List[Dict[str, Any]],
    t: float,
    profile: Optional[Dict[str, Any]] = None,
) -> np.ndarray:

    canvas = background.copy()

    H, W = canvas.shape[:2]

    stride_mult = 1.0

    if profile:

        try:
            stride_mult = float(
                profile.get(
                    "stride_multiplier",
                    1.0,
                )
            )
        except Exception:
            stride_mult = 1.0

    body_bob = (
        math.sin(
            2.0
            * math.pi
            * t
        )
        * H
        * BODY_BOB
    )

    # --------------------------------------------------------
    # IMPORTANT:
    # LEGS FIRST
    # BODY LAST
    #
    # This prevents the original stationary legs from being
    # visible beneath the moving legs.
    # --------------------------------------------------------

    for index, leg in enumerate(
        prepared_legs
    ):

        side = str(
            leg.get(
                "side",
                "",
            )
        ).lower()

        phase = phase_for(
            side,
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

        angle = (
            gait
            * STEP_ANGLE
            * stride_mult
        )

        if "back" in side:
            angle *= 0.90

        rotated, x, y = swing_leg(
            leg,
            angle,
        )

        alpha_over(
            canvas,
            rotated,
            x,
            int(y + body_bob),
        )

    # --------------------------------------------------------
    # BODY LAST
    # --------------------------------------------------------

    bx, by, _, _ = body_bbox

    alpha_over(
        canvas,
        body_sprite,
        bx,
        int(by + body_bob),
    )

    return canvas


# ============================================================
# STANDING POSE
# ============================================================

def create_standing_pose(
    background: np.ndarray,
    body_sprite: np.ndarray,
    body_bbox: Tuple[int, int, int, int],
    prepared_legs: List[Dict[str, Any]],
) -> np.ndarray:

    canvas = background.copy()

    for leg in prepared_legs:

        sprite = leg["sprite"]

        x, y, _, _ = leg["bbox"]

        alpha_over(
            canvas,
            sprite,
            x,
            y,
        )

    bx, by, _, _ = body_bbox

    alpha_over(
        canvas,
        body_sprite,
        bx,
        by,
    )

    return canvas


# ============================================================
# EASING
# ============================================================

def ease_in_out(
    t: float,
) -> float:

    t = float(
        np.clip(
            t,
            0.0,
            1.0,
        )
    )

    return (
        t * t
        * (3.0 - 2.0 * t)
    )


# ============================================================
# TRANSLATE COMPLETE WALKING POSE
# ============================================================

def translate_walking_pose(
    pose: np.ndarray,
    dx: int,
    dy: int,
) -> np.ndarray:

    H, W = pose.shape[:2]

    matrix = np.float32([
        [1, 0, dx],
        [0, 1, dy],
    ])

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
    white_canvas: np.ndarray,
    body_sprite: np.ndarray,
    body_bbox: Tuple[int, int, int, int],
    prepared_legs: List[Dict[str, Any]],
    t: float,
    travel_p: float,
    profile: Optional[Dict[str, Any]] = None,
) -> np.ndarray:

    H, W = white_canvas.shape[:2]

    pose = create_walking_pose(
        white_canvas,
        body_sprite,
        body_bbox,
        prepared_legs,
        t,
        profile,
    )

    bx = body_bbox[0]

    animal_width = max(
        1,
        body_bbox[2]
        - body_bbox[0],
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

    target_x = bx

    p = ease_in_out(
        travel_p
    )

    current_x = (
        start_x
        + (
            target_x
            - start_x
        )
        * p
    )

    dx = int(
        round(
            current_x - bx
        )
    )

    dy = int(
        round(
            math.sin(
                travel_p
                * math.pi
            )
            * H
            * 0.004
        )
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
    animated_frame: np.ndarray,
    original: np.ndarray,
    p: float,
) -> np.ndarray:

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
    total_frames: int,
    mode: str,
    walk_fraction: float,
    stand_fraction: float,
    merge_fraction: float,
) -> Tuple[int, int, int]:

    if mode == "Walk in place only":
        return (
            0,
            total_frames,
            0,
        )

    walk_n = max(
        1,
        int(
            round(
                total_frames
                * walk_fraction
            )
        ),
    )

    merge_n = 0

    if (
        mode
        == "White canvas → walk in → stand → merge"
    ):
        merge_n = max(
            1,
            int(
                round(
                    total_frames
                    * merge_fraction
                )
            ),
        )

    stand_n = (
        total_frames
        - walk_n
        - merge_n
    )

    if stand_n < 1:

        stand_n = 1

        remaining = (
            total_frames
            - stand_n
        )

        if merge_n > 0:

            merge_n = min(
                merge_n,
                max(
                    1,
                    remaining
                    // 3,
                ),
            )

        walk_n = (
            total_frames
            - stand_n
            - merge_n
        )

        walk_n = max(
            1,
            walk_n,
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
    original: np.ndarray,
    body_sprite: np.ndarray,
    body_bbox: Tuple[int, int, int, int],
    prepared_legs: List[Dict[str, Any]],
    total_frames: int,
    profile: Optional[Dict[str, Any]] = None,
) -> List[np.ndarray]:

    H, W = original.shape[:2]

    white_canvas = np.ones(
        (H, W, 3),
        dtype=np.uint8,
    ) * 255

    walk_n, stand_n, merge_n = (
        allocate_frames(
            total_frames,
            ANIMATION_MODE,
            WALK_IN_FRACTION,
            STAND_FRACTION,
            MERGE_FRACTION,
        )
    )

    frames = []

    # ========================================================
    # PHASE 0 — PURE WHITE INTRO
    # ========================================================

    intro_n = min(
        3,
        max(
            0,
            walk_n - 1,
        ),
    )

    for _ in range(intro_n):
        frames.append(
            white_canvas.copy()
        )

    # ========================================================
    # PHASE 1 — WALK IN
    # ========================================================

    actual_walk_n = (
        walk_n - intro_n
    )

    if actual_walk_n > 0:

        cycles = max(
            1,
            WALK_CYCLES,
        )

        for i in range(
            actual_walk_n
        ):

            if actual_walk_n == 1:
                travel_p = 1.0
            else:
                travel_p = (
                    i
                    / float(
                        actual_walk_n
                        - 1
                    )
                )

            gait_t = (
                travel_p
                * cycles
            )

            frame = (
                create_moving_walking_frame(
                    white_canvas,
                    body_sprite,
                    body_bbox,
                    prepared_legs,
                    gait_t,
                    travel_p,
                    profile,
                )
            )

            frames.append(
                frame
            )

    # ========================================================
    # PHASE 2 — STAND / WALK IN PLACE
    # ========================================================

    if stand_n > 0:

        if WALK_CYCLES <= 0:

            standing = (
                create_standing_pose(
                    white_canvas,
                    body_sprite,
                    body_bbox,
                    prepared_legs,
                )
            )

            for _ in range(
                stand_n
            ):
                frames.append(
                    standing.copy()
                )

        else:

            for i in range(
                stand_n
            ):

                if stand_n == 1:
                    t = 0.0
                else:
                    t = (
                        i
                        / float(
                            stand_n
                            - 1
                        )
                    )

                gait_t = (
                    t
                    * max(
                        1,
                        WALK_CYCLES,
                    )
                )

                frame = (
                    create_walking_pose(
                        white_canvas,
                        body_sprite,
                        body_bbox,
                        prepared_legs,
                        gait_t,
                        profile,
                    )
                )

                frames.append(
                    frame
                )

    # ========================================================
    # FINAL ISOLATED ANIMAL FRAME
    # ========================================================

    if not frames:

        frames.append(
            create_standing_pose(
                white_canvas,
                body_sprite,
                body_bbox,
                prepared_legs,
            )
        )

    isolated_frame = (
        frames[-1].copy()
    )

    # ========================================================
    # PHASE 3 — MERGE INTO ORIGINAL SCENERY
    # ========================================================

    if merge_n > 0:

        for i in range(
            merge_n
        ):

            if merge_n == 1:
                p = 1.0
            else:
                p = (
                    i
                    / float(
                        merge_n
                        - 1
                    )
                )

            frames.append(
                smooth_merge(
                    isolated_frame,
                    original,
                    p,
                )
            )

    # ========================================================
    # FORCE EXACT FRAME COUNT
    # ========================================================

    if len(frames) > total_frames:

        frames = frames[
            :total_frames
        ]

    while len(frames) < total_frames:

        frames.append(
            frames[-1].copy()
        )

    # ========================================================
    # FINAL FRAME MUST BE EXACT ORIGINAL
    # ========================================================

    if (
        ANIMATION_MODE
        == "White canvas → walk in → stand → merge"
    ):
        frames[-1] = original.copy()

    return frames


# ============================================================
# DETECTION OVERLAY
# ============================================================

def detection_overlay(
    image: np.ndarray,
    scene: Dict[str, Any],
) -> np.ndarray:

    overlay = image.copy()

    h, w = image.shape[:2]

    animal_polygon = (
        scene.get(
            "animal_polygon"
        )
        or []
    )

    pts = poly_px(
        animal_polygon,
        w,
        h,
    )

    if len(pts) >= 3:

        cv2.polylines(
            overlay,
            [pts],
            True,
            (0, 255, 0),
            3,
        )

    for part in scene.get(
        "parts",
        [],
    ):

        if not isinstance(
            part,
            dict,
        ):
            continue

        polygon = (
            part.get(
                "polygon",
                [],
            )
            or []
        )

        pts = poly_px(
            polygon,
            w,
            h,
        )

        if len(pts) >= 3:

            cv2.polylines(
                overlay,
                [pts],
                True,
                (0, 165, 255),
                2,
            )

        joints = (
            part.get(
                "joints",
                {},
            )
            or {}
        )

        for joint_name in (
            "proximal",
            "middle",
            "distal",
        ):

            if joint_name not in joints:
                continue

            x, y = pt_px(
                joints[joint_name],
                w,
                h,
            )

            cv2.circle(
                overlay,
                (x, y),
                6,
                (0, 0, 255),
                -1,
            )

    return overlay


# ============================================================
# GIF EXPORT
# ============================================================

def gif_bytes(
    frames: List[np.ndarray],
    fps: int,
) -> bytes:

    if not frames:
        return b""

    pil_frames = []

    for frame in frames:

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB,
        )

        pil_frames.append(
            Image.fromarray(rgb)
        )

    buffer = io.BytesIO()

    duration = int(
        1000
        / max(
            1,
            fps,
        )
    )

    pil_frames[0].save(
        buffer,
        format="GIF",
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration,
        loop=0,
    )

    return buffer.getvalue()


# ============================================================
# MP4 EXPORT
# ============================================================

def mp4_bytes(
    frames: List[np.ndarray],
    fps: int,
) -> bytes:

    if not frames:
        return b""

    H, W = frames[0].shape[:2]

    temp_file = tempfile.NamedTemporaryFile(
        suffix=".mp4",
        delete=False,
    )

    temp_path = temp_file.name
    temp_file.close()

    try:

        writer = cv2.VideoWriter(
            temp_path,
            cv2.VideoWriter_fourcc(
                *"mp4v"
            ),
            fps,
            (W, H),
        )

        for frame in frames:
            writer.write(frame)

        writer.release()

        with open(
            temp_path,
            "rb",
        ) as f:
            data = f.read()

        return data

    finally:

        try:
            os.remove(
                temp_path
            )
        except Exception:
            pass


# ============================================================
# ZIP FRAMES
# ============================================================

def zip_frames(
    frames: List[np.ndarray],
    prefix: str = "frame",
) -> bytes:

    buffer = io.BytesIO()

    with zipfile.ZipFile(
        buffer,
        "w",
        zipfile.ZIP_DEFLATED,
    ) as archive:

        for i, frame in enumerate(
            frames
        ):

            success, encoded = (
                cv2.imencode(
                    ".png",
                    frame,
                )
            )

            if not success:
                continue

            archive.writestr(
                f"{prefix}_{i + 1:04d}.png",
                encoded.tobytes(),
            )

    return buffer.getvalue()


# ============================================================
# UI — UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "Upload your hand-drawn animal image",
    type=[
        "png",
        "jpg",
        "jpeg",
    ],
)

if uploaded_file is None:

    st.info(
        "Upload a hand-drawn animal image to begin."
    )

    st.stop()


# ============================================================
# READ IMAGE
# ============================================================

try:

    uploaded_bytes = uploaded_file.read()

    image_pil = Image.open(
        io.BytesIO(
            uploaded_bytes
        )
    ).convert("RGB")

    image_bgr = cv2.cvtColor(
        np.array(image_pil),
        cv2.COLOR_RGB2BGR,
    )

except Exception as exc:

    st.error(
        f"Could not read uploaded image: {exc}"
    )

    st.stop()


# PNG for Gemini
gemini_buffer = io.BytesIO()

image_pil.save(
    gemini_buffer,
    format="PNG",
)

gemini_bytes = (
    gemini_buffer.getvalue()
)


# ============================================================
# ORIGINAL IMAGE
# ============================================================

c1, c2 = st.columns(2)

with c1:

    st.image(
        cv2.cvtColor(
            image_bgr,
            cv2.COLOR_BGR2RGB,
        ),
        caption="Original drawing",
        width="stretch",
    )

with c2:

    st.markdown(
        """
### 🧠 Animation pipeline

1. Gemini identifies the animal anatomy.
2. Python extracts the ORIGINAL animal pixels.
3. Every visible leg is isolated.
4. Proximal joints are used as stable pivots.
5. The entire assembled animal is moved together.
6. The original stationary animal is hidden during walking.
7. The animal walks in from outside the frame.
8. It reaches its exact original position.
9. It continues walking in place.
10. The animated animal slowly merges into the original scenery.
        """
    )


# ============================================================
# STEP 1 — GEMINI ANALYSIS
# ============================================================

if st.button(
    "🔍 Analyze drawing with Gemini",
    type="primary",
    width="stretch",
):

    with st.spinner(
        "Analyzing animal anatomy..."
    ):

        scene = analyze_scene(
            gemini_bytes,
            GEMINI_API_KEY,
        )

    if scene:

        st.session_state.scene = scene

        st.session_state.walk_keyframes = None

        st.session_state.frames = None


# ============================================================
# STOP UNTIL ANALYSIS EXISTS
# ============================================================

if not st.session_state.scene:

    st.stop()


scene = st.session_state.scene


# ============================================================
# DETECTION RESULT
# ============================================================

st.success(
    f"Detected: "
    f"**{scene.get('identified_character', 'character')}**"
)


with st.expander(
    "🧬 Gemini anatomy JSON"
):

    st.json(scene)


overlay = detection_overlay(
    image_bgr,
    scene,
)

st.image(
    cv2.cvtColor(
        overlay,
        cv2.COLOR_BGR2RGB,
    ),
    caption=(
        "Detection: green = animal, "
        "orange = leg polygons, "
        "red = joints"
    ),
    width="stretch",
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
        image_bgr,
        part,
    )

    if prepared is not None:

        prepared_legs.append(
            prepared
        )


if not prepared_legs:

    st.error(
        "No usable leg polygons/joints were returned. "
        "Try a clearer image or analyze again."
    )

    st.stop()


st.success(
    f"Prepared "
    f"**{len(prepared_legs)}** "
    f"original leg cut-outs."
)


# ============================================================
# EXTRACT COMPLETE ANIMAL
# ============================================================

animal, animal_bbox, animal_mask = (
    None,
    None,
    None,
)

animal_polygon = (
    scene.get(
        "animal_polygon"
    )
    or []
)

if len(animal_polygon) >= 3:

    h, w = image_bgr.shape[:2]

    animal_mask = poly_mask(
        (h, w),
        animal_polygon,
        dilation=max(
            1,
            min(h, w) // 500,
        ),
    )

    animal_bbox = bbox_poly(
        animal_polygon,
        w,
        h,
        margin=max(
            8,
            min(h, w) // 150,
        ),
    )

else:

    h, w = image_bgr.shape[:2]

    animal_bbox = bbox_norm(
        scene.get(
            "animal_bbox"
        ),
        w,
        h,
        margin=max(
            8,
            min(h, w) // 150,
        ),
    )

    animal_mask = np.zeros(
        (h, w),
        dtype=np.uint8,
    )

    x1, y1, x2, y2 = animal_bbox

    animal_mask[
        y1:y2,
        x1:x2
    ] = 255


x1, y1, x2, y2 = animal_bbox

animal_crop = image_bgr[
    y1:y2,
    x1:x2
].copy()

animal_alpha = animal_mask[
    y1:y2,
    x1:x2
].copy()

animal = cv2.cvtColor(
    animal_crop,
    cv2.COLOR_BGR2BGRA,
)

animal[:, :, 3] = animal_alpha


# ============================================================
# BODY EXTRACTION
# ============================================================

body_sprite = make_body_sprite(
    animal,
    animal_bbox,
    prepared_legs,
)


# ============================================================
# WALKING PROFILE
# ============================================================

profile = (
    scene.get(
        "locomotion_profile"
    )
    or {}
)


# ============================================================
# STEP 2 — CONNECTED WALKING POSES
# ============================================================

st.markdown("---")

st.header(
    "🦵 Step 2 — Connected walking poses"
)

st.write(
    "The complete animated animal is assembled from the "
    "original drawing. The legs move around their anatomical "
    "proximal pivots while the body is rendered last."
)


if st.button(
    "🦒 Generate 4 connected walking poses",
    type="primary",
    width="stretch",
):

    with st.spinner(
        "Building connected walking poses..."
    ):

        poses = []

        white = np.ones_like(
            image_bgr
        ) * 255

        for t in (
            0.00,
            0.25,
            0.50,
            0.75,
        ):

            pose = create_walking_pose(
                white,
                body_sprite,
                animal_bbox,
                prepared_legs,
                t,
                profile,
            )

            poses.append(
                pose
            )

        st.session_state.walk_keyframes = poses

        st.session_state.frames = None


# ============================================================
# DISPLAY KEY POSES
# ============================================================

if st.session_state.walk_keyframes:

    cols = st.columns(4)

    for i, frame in enumerate(
        st.session_state.walk_keyframes
    ):

        with cols[i]:

            st.image(
                cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB,
                ),
                caption=f"Pose {i + 1}",
                width="stretch",
            )

    st.download_button(
        "⬇️ Download 4 walking poses",
        zip_frames(
            st.session_state.walk_keyframes,
            "walk_pose",
        ),
        "animal_walk_keyposes.zip",
        "application/zip",
        width="stretch",
    )


# ============================================================
# STEP 3 — RENDER COMPLETE ANIMATION
# ============================================================

st.markdown("---")

st.header(
    "🎞️ Step 3 — Render complete animation"
)


if not st.session_state.walk_keyframes:

    st.warning(
        "Generate the 4 connected walking poses first."
    )

else:

    if st.button(
        "🚀 Render complete animation",
        type="primary",
        width="stretch",
    ):

        with st.spinner(
            "Rendering animation..."
        ):

            frames = build_animation(
                image_bgr,
                body_sprite,
                animal_bbox,
                prepared_legs,
                TOTAL_FRAMES,
                profile,
            )

            st.session_state.frames = frames

        st.success(
            f"Rendered "
            f"**{len(frames)} frames** "
            f"at **{FPS} FPS**."
        )


# ============================================================
# RESULT
# ============================================================

if st.session_state.frames:

    frames = st.session_state.frames

    st.markdown("---")

    st.header(
        "🎬 Final animation"
    )

    gif_data = gif_bytes(
        frames,
        FPS,
    )

    st.image(
        gif_data,
        caption="Animated GIF",
        width="stretch",
    )

    # --------------------------------------------------------
    # DOWNLOAD GIF
    # --------------------------------------------------------

    st.download_button(
        "⬇️ Download GIF",
        gif_data,
        "animal_walk_animation.gif",
        "image/gif",
        width="stretch",
    )

    # --------------------------------------------------------
    # DOWNLOAD MP4
    # --------------------------------------------------------

    with st.spinner(
        "Preparing MP4..."
    ):

        mp4_data = mp4_bytes(
            frames,
            FPS,
        )

    st.download_button(
        "⬇️ Download MP4",
        mp4_data,
        "animal_walk_animation.mp4",
        "video/mp4",
        width="stretch",
    )

    # --------------------------------------------------------
    # DOWNLOAD PNG FRAMES
    # --------------------------------------------------------

    png_zip = zip_frames(
        frames,
        "animal_frame",
    )

    st.download_button(
        "⬇️ Download PNG frames",
        png_zip,
        "animal_animation_frames.zip",
        "application/zip",
        width="stretch",
    )

    # --------------------------------------------------------
    # FRAME PREVIEW
    # --------------------------------------------------------

    st.markdown("---")

    st.subheader(
        "🖼️ Frame preview"
    )

    preview_count = min(
        16,
        len(frames),
    )

    preview_indices = np.linspace(
        0,
        len(frames) - 1,
        preview_count,
        dtype=int,
    )

    preview_cols = st.columns(4)

    for i, frame_index in enumerate(
        preview_indices
    ):

        with preview_cols[
            i % 4
        ]:

            st.image(
                cv2.cvtColor(
                    frames[frame_index],
                    cv2.COLOR_BGR2RGB,
                ),
                caption=(
                    f"Frame "
                    f"{frame_index + 1}"
                ),
                width="stretch",
            )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "🦒 Solid Connected Walk v8 — "
    "Original-pixel anatomy, connected leg pivots, "
    "anti-ghosting body extraction, slow walk-in, "
    "and gradual scenery merge."
)

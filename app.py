import io
import json
import math
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

st.title("🦒 Hand-Drawn Animal Walk Animator")
st.caption(
    "Gemini identifies the animal and its anatomy. Python creates a "
    "4-pose walking cycle from the ORIGINAL drawing with precision annotations."
)

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("⚙️ Settings")

GEMINI_API_KEY = st.sidebar.text_input(
    "Gemini API Key",
    type="password",
    help="Your Google Gemini API key.",
)

MODEL_SELECTION = st.sidebar.selectbox(
    "Gemini model",
    [
        "AUTO — use an available vision model",
    ],
    index=0,
    help=(
        "The app automatically asks your Gemini API key which models are "
        "available, then tests them and uses the first working model. "
        "No obsolete model names are hard-coded as fallbacks."
    ),
)

st.sidebar.markdown("---")

ANIMATION_MODE = st.sidebar.selectbox(
    "Animation mode",
    [
        "Walk in → walk in place → merge",
        "Walk in → walk in place",
        "Walk in place only",
    ],
)

TOTAL_FRAMES = st.sidebar.slider(
    "Total animation frames",
    20,
    100,
    48,
    2,
)

FPS = st.sidebar.slider(
    "FPS",
    4,
    20,
    8,
)

WALK_CYCLES = st.sidebar.slider(
    "Walking cycles",
    1,
    5,
    2,
)

STRIDE = st.sidebar.slider(
    "Leg stride",
    0.05,
    0.35,
    0.16,
    0.01,
    help="How far the legs swing. Start low for delicate hand-drawn animals.",
)

BOB_AMOUNT = st.sidebar.slider(
    "Body bob",
    0.0,
    0.05,
    0.012,
    0.002,
)

WALK_IN_FRACTION = st.sidebar.slider(
    "Walk-in portion",
    0.05,
    0.45,
    0.20,
    0.05,
)

MERGE_FRACTION = st.sidebar.slider(
    "Final merge portion",
    0.05,
    0.35,
    0.15,
    0.05,
)

st.sidebar.markdown("---")
st.sidebar.info(
    "Tip: use a clean scan/photo of the drawing. The better Gemini can "
    "see the animal's legs, the better the automatic walking cycle."
)


# ============================================================
# GEMINI ANALYSIS & DISCOVERY
# ============================================================

def clean_json_text(text: str) -> str:
    """Remove accidental markdown fences around JSON."""
    text = (text or "").strip()

    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    return text


def _model_name(model_obj: Any) -> str:
    """Return a plain model name from a Gemini SDK model object."""
    name = getattr(model_obj, "name", "") or ""
    return str(name).strip()


def _model_actions(model_obj: Any) -> List[str]:
    """Return supported actions without assuming a specific SDK version."""
    actions = getattr(model_obj, "supported_actions", None)
    if actions is None:
        actions = getattr(model_obj, "supportedActions", None)
    if actions is None:
        return []
    try:
        return [str(a) for a in actions]
    except Exception:
        return []


def discover_available_models(client: genai.Client) -> List[str]:
    """Ask the Gemini API which models are actually available for this API key."""
    discovered: List[str] = []
    try:
        for model_obj in client.models.list():
            name = _model_name(model_obj)
            if not name:
                continue

            actions = _model_actions(model_obj)
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


def model_priority_score(name: str) -> Tuple[int, str]:
    """Rank currently available models for image understanding."""
    n = name.lower().replace("models/", "")

    if "gemini-3.7-flash" in n:
        return (0, n)
    if "gemini-3.6-flash" in n:
        return (1, n)
    if "gemini-3.5-flash" in n:
        return (2, n)
    if "gemini-3.1-flash" in n and "lite" not in n:
        return (3, n)
    if "gemini-3.1-flash-lite" in n:
        return (4, n)
    if "gemini-3" in n and "flash" in n:
        return (5, n)
    if "gemini-2.5-flash" in n and "lite" not in n:
        return (6, n)
    if "gemini-2.5-flash-lite" in n:
        return (7, n)
    if "gemini-2.5" in n:
        return (8, n)
    if "gemini-2" in n and "flash" in n:
        return (10, n)
    if "gemini" in n and "pro" in n:
        return (15, n)
    if "gemini" in n and "flash" in n:
        return (20, n)
    if "gemini" in n:
        return (30, n)
    return (100, n)


def ordered_model_candidates(client: genai.Client) -> List[str]:
    """Return dynamically available generateContent models in best-first order."""
    available = discover_available_models(client)

    gemini_models = [m for m in available if "gemini" in m.lower()]
    other_models = [m for m in available if "gemini" not in m.lower()]

    gemini_models.sort(key=model_priority_score)
    other_models.sort(key=model_priority_score)

    return gemini_models + other_models


def _safe_response_text(response: Any) -> str:
    """Extract response text across small SDK response-shape differences."""
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


def analyze_and_segment_scene(
    image_bytes: bytes,
    api_key: str,
    model_name: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Analyze the drawing using the updated precision anatomy prompt."""
    if not api_key:
        st.error("Please enter a Gemini API key.")
        return None

    try:
        client = genai.Client(api_key=api_key)
    except Exception as exc:
        st.error(f"Could not initialize Gemini: {exc}")
        return None

    prompt = r"""
You are a precision visual-anatomy annotator for a hand-drawn 2D animation
system.

You are analyzing ONE hand-drawn animal image.

IMPORTANT:
This image will NOT be redrawn by AI.

Python will physically cut pixels from the ORIGINAL IMAGE and move them.
Therefore your coordinates must describe the actual visible pixels as
accurately as possible.

============================================================
PRIMARY TASK
============================================================

Identify the animal and identify EVERY VISIBLE LEG.

For every visible leg:

1. Trace ONLY that leg.
2. Do NOT include background.
3. Do NOT include the body.
4. Do NOT include another leg.
5. Do NOT include shadows on the floor.
6. Do NOT include large empty regions surrounding the leg.
7. Follow the visible outer boundary of the drawn leg.
8. Include the black outline belonging to the leg.
9. Include the colored interior belonging to the leg.
10. Include the hoof/foot.
11. The polygon must begin at the actual attachment point to the body.
12. Do not cut through the leg unnecessarily.

============================================================
VERY IMPORTANT: LEG ANATOMY
============================================================

For each visible leg identify:

root
upper_leg
middle_joint
lower_leg
hoof

For a quadruped, estimate:

proximal joint = shoulder/hip region
middle joint = knee/elbow region
distal joint = ankle/wrist/hoof region

Do NOT place the middle joint in the middle of the bounding box simply
because it is convenient.

Place it where the drawn leg actually changes direction.

============================================================
OVERLAPPING LEGS
============================================================

If one leg overlaps another:

- identify the visible leg separately
- do NOT merge the two legs into one polygon
- do NOT invent the hidden portion
- use only pixels that are actually visible

If the boundary between two legs is uncertain, prefer a smaller polygon
rather than accidentally including pixels from the other leg.

============================================================
BODY CONNECTION
============================================================

The top/root of the leg is extremely important.

The polygon should touch the body exactly where the leg emerges.

Do NOT include a large piece of the abdomen/body.

Do NOT make the leg polygon rectangular.

Do NOT make a large triangular polygon around the leg.

============================================================
JOINT COORDINATES
============================================================

Coordinates use normalized [y,x] format from 0 to 100.

Provide:

root
proximal
middle
distal
hoof

The middle joint must be positioned on the visible centerline of the
drawn leg.

============================================================
LEG CENTERLINE
============================================================

Also provide a centerline containing 5-9 points following the actual
center of the visible leg.

The centerline must begin at the body attachment and end at the hoof.

============================================================
LEG WIDTH
============================================================

Estimate the visible width of the leg at several points.

Return:

width_profile:
[
    {"at":0.0,"width":...},
    {"at":0.25,"width":...},
    {"at":0.50,"width":...},
    {"at":0.75,"width":...},
    {"at":1.0,"width":...}
]

These widths are normalized relative to image width.

============================================================
BODY
============================================================

Identify the body separately.

The body polygon should exclude the legs as much as reasonably possible.

============================================================
OUTPUT
============================================================

Return ONLY valid JSON.

Use this exact structure:

{
  "identified_character": "giraffe",

  "animal_bbox": [ymin,xmin,ymax,xmax],

  "animal_polygon": [
    [y,x],
    ...
  ],

  "parts": [

    {
      "name": "body",
      "type": "body",
      "polygon": [
        [y,x],
        ...
      ]
    },

    {
      "name": "front_left_leg",
      "type": "leg",
      "side": "front_left",

      "polygon": [
        [y,x],
        ...
      ],

      "joints": {
        "root": [y,x],
        "proximal": [y,x],
        "middle": [y,x],
        "distal": [y,x],
        "hoof": [y,x]
      },

      "centerline": [
        [y,x],
        [y,x],
        [y,x],
        [y,x],
        [y,x]
      ],

      "width_profile": [
        {"at":0.0,"width":0.0},
        {"at":0.25,"width":0.0},
        {"at":0.50,"width":0.0},
        {"at":0.75,"width":0.0},
        {"at":1.0,"width":0.0}
      ]
    }
  ],

  "notes": ""
}

============================================================
QUALITY CONTROL BEFORE RETURNING JSON
============================================================

Before returning the answer, mentally inspect every leg.

For EACH leg ask:

A. Does the polygon contain only the leg?
B. Does it include the complete visible hoof?
C. Does it accidentally contain background?
D. Does it accidentally contain body pixels?
E. Does it accidentally contain another leg?
F. Is the root located at the real body attachment?
G. Is the middle joint located at the actual bend?
H. Does the centerline follow the actual drawn leg?

If any answer is wrong, correct the coordinates before returning JSON.

Do NOT simplify the leg into a rectangle.

Do NOT approximate the leg using a generic animal model.

Use the actual pixels visible in THIS image.
"""

    discovered = ordered_model_candidates(client)

    if not discovered:
        st.error(
            "Gemini API did not return any models supporting generateContent. "
            "Check that the API key is valid and that the Gemini API is enabled."
        )
        return None

    candidates: List[str] = []
    if model_name:
        requested = model_name.strip()
        if requested and requested.lower().startswith("auto"):
            requested = ""
        if requested:
            for available in discovered:
                if available.lower() == requested.lower() or available.lower().endswith(requested.lower()):
                    candidates.append(available)
                    break

    for candidate in discovered:
        if candidate not in candidates:
            candidates.append(candidate)

    st.info(
        f"🔎 Gemini discovered {len(discovered)} generateContent model(s). "
        f"Testing the best available model first: `{candidates[0]}`"
    )

    errors: List[str] = []

    for current_model in candidates:
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
                        temperature=0.1,
                    ),
                )
            except Exception:
                response = client.models.generate_content(
                    model=current_model,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        temperature=0.1,
                    ),
                )

            text = clean_json_text(_safe_response_text(response))
            if not text:
                raise ValueError("The model returned an empty response.")

            data = json.loads(text)
            if not isinstance(data, dict):
                raise ValueError("The model response was not a JSON object.")

            st.success(f"✅ Analysis completed with `{current_model}`")
            return data

        except Exception as exc:
            error_text = str(exc).replace("\n", " ")
            errors.append(f"{current_model}: {error_text}")
            continue

    st.error("Gemini analysis failed after trying every available model.")
    with st.expander("Show model attempts"):
        for error in errors:
            st.code(error)
    return None


# ============================================================
# GEOMETRY HELPERS
# ============================================================

def clamp_point(point: List[float]) -> Tuple[float, float]:
    y = float(np.clip(point[0], 0, 100))
    x = float(np.clip(point[1], 0, 100))
    return y, x


def normalized_point_to_px(
    point: List[float],
    width: int,
    height: int,
) -> Tuple[int, int]:
    y, x = clamp_point(point)
    return int(x * width / 100.0), int(y * height / 100.0)


def normalized_polygon_to_px(
    polygon: List[List[float]],
    width: int,
    height: int,
) -> np.ndarray:
    pts = []
    for p in polygon:
        if isinstance(p, (list, tuple)) and len(p) >= 2:
            x, y = normalized_point_to_px(p, width, height)
            pts.append([x, y])

    if len(pts) < 3:
        return np.empty((0, 2), dtype=np.int32)

    return np.asarray(pts, dtype=np.int32)


def polygon_mask(
    shape: Tuple[int, int],
    polygon: List[List[float]],
    dilation: int = 0,
) -> np.ndarray:
    h, w = shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    pts = normalized_polygon_to_px(polygon, w, h)

    if len(pts) >= 3:
        cv2.fillPoly(mask, [pts], 255)

    if dilation > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (dilation * 2 + 1, dilation * 2 + 1),
        )
        mask = cv2.dilate(mask, kernel)

    return mask


def bbox_from_polygon(
    polygon: List[List[float]],
    width: int,
    height: int,
    margin: int = 10,
) -> Tuple[int, int, int, int]:

    pts = normalized_polygon_to_px(polygon, width, height)

    if len(pts) == 0:
        return 0, 0, width, height

    x, y, bw, bh = cv2.boundingRect(pts)

    x1 = max(0, x - margin)
    y1 = max(0, y - margin)
    x2 = min(width, x + bw + margin)
    y2 = min(height, y + bh + margin)

    return x1, y1, x2, y2


def bbox_from_normalized(
    bbox: Optional[List[float]],
    width: int,
    height: int,
    margin: int = 10,
) -> Tuple[int, int, int, int]:

    if not bbox or len(bbox) != 4:
        return 0, 0, width, height

    ymin, xmin, ymax, xmax = [float(v) for v in bbox]

    x1 = int(xmin * width / 100)
    y1 = int(ymin * height / 100)
    x2 = int(xmax * width / 100)
    y2 = int(ymax * height / 100)

    x1 = max(0, x1 - margin)
    y1 = max(0, y1 - margin)
    x2 = min(width, x2 + margin)
    y2 = min(height, y2 + margin)

    return x1, y1, x2, y2


# ============================================================
# IMAGE / ALPHA HELPERS
# ============================================================

def rgba_from_masked_crop(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    bbox: Tuple[int, int, int, int],
) -> np.ndarray:

    x1, y1, x2, y2 = bbox

    crop = image_bgr[y1:y2, x1:x2].copy()
    crop_mask = mask[y1:y2, x1:x2].copy()

    if crop.size == 0:
        return np.zeros((1, 1, 4), dtype=np.uint8)

    alpha = cv2.GaussianBlur(crop_mask, (3, 3), 0)

    rgba = cv2.cvtColor(crop, cv2.COLOR_BGR2BGRA)
    rgba[:, :, 3] = alpha

    return rgba


def extract_animal(
    image_bgr: np.ndarray,
    scene_data: Dict[str, Any],
) -> Tuple[np.ndarray, Tuple[int, int, int, int], np.ndarray]:

    h, w = image_bgr.shape[:2]

    polygon = scene_data.get("animal_polygon") or []

    if len(polygon) >= 3:
        mask = polygon_mask((h, w), polygon, dilation=max(1, min(h, w) // 500))
        bbox = bbox_from_polygon(polygon, w, h, margin=max(8, min(h, w) // 150))
    else:
        bbox = bbox_from_normalized(
            scene_data.get("animal_bbox"),
            w,
            h,
            margin=max(8, min(h, w) // 150),
        )
        mask = np.zeros((h, w), dtype=np.uint8)
        x1, y1, x2, y2 = bbox
        mask[y1:y2, x1:x2] = 255

    kernel_size = max(3, int(min(h, w) / 250) * 2 + 1)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_size, kernel_size),
    )
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    x1, y1, x2, y2 = bbox
    crop = image_bgr[y1:y2, x1:x2].copy()
    crop_mask = mask[y1:y2, x1:x2]

    rgba = cv2.cvtColor(crop, cv2.COLOR_BGR2BGRA)
    rgba[:, :, 3] = crop_mask

    return rgba, bbox, mask


def overlay_rgba(
    canvas_bgr: np.ndarray,
    sprite_rgba: np.ndarray,
    x: int,
    y: int,
    opacity: float = 1.0,
) -> np.ndarray:

    if sprite_rgba is None or sprite_rgba.size == 0:
        return canvas_bgr

    sh, sw = sprite_rgba.shape[:2]
    H, W = canvas_bgr.shape[:2]

    x2 = x + sw
    y2 = y + sh

    if x2 <= 0 or y2 <= 0 or x >= W or y >= H:
        return canvas_bgr

    cx1 = max(0, x)
    cy1 = max(0, y)
    cx2 = min(W, x2)
    cy2 = min(H, y2)

    sx1 = cx1 - x
    sy1 = cy1 - y
    sx2 = sx1 + (cx2 - cx1)
    sy2 = sy1 + (cy2 - cy1)

    src = sprite_rgba[sy1:sy2, sx1:sx2]

    alpha = (
        src[:, :, 3].astype(np.float32) / 255.0
    )[:, :, None] * float(opacity)

    src_rgb = src[:, :, :3].astype(np.float32)
    dst = canvas_bgr[cy1:cy2, cx1:cx2].astype(np.float32)

    result = src_rgb * alpha + dst * (1.0 - alpha)

    canvas_bgr[cy1:cy2, cx1:cx2] = np.clip(
        result,
        0,
        255,
    ).astype(np.uint8)

    return canvas_bgr


def rotate_rgba(
    sprite: np.ndarray,
    angle: float,
    center: Optional[Tuple[float, float]] = None,
) -> Tuple[np.ndarray, np.ndarray]:

    h, w = sprite.shape[:2]

    if center is None:
        center = (w / 2, h / 2)

    M = cv2.getRotationMatrix2D(
        center,
        angle,
        1.0,
    )

    cos = abs(M[0, 0])
    sin = abs(M[0, 1])

    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)

    M[0, 2] += new_w / 2 - center[0]
    M[1, 2] += new_h / 2 - center[1]

    rotated = cv2.warpAffine(
        sprite,
        M,
        (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )

    return rotated, M


# ============================================================
# LEG PREPARATION
# ============================================================

def prepare_leg_sprite(
    image_bgr: np.ndarray,
    leg: Dict[str, Any],
) -> Optional[Dict[str, Any]]:

    polygon = leg.get("polygon") or []
    joints = leg.get("joints") or {}

    if len(polygon) < 3:
        return None

    # Support both new precision keys ('proximal' or fallback check)
    if not isinstance(joints, dict) or not any(k in joints for k in ("proximal", "root")):
        return None

    # Normalize joints lookup
    joint_target = joints.get("proximal") or joints.get("root")

    h, w = image_bgr.shape[:2]

    mask = polygon_mask(
        (h, w),
        polygon,
        dilation=max(2, min(h, w) // 250),
    )

    x1, y1, x2, y2 = bbox_from_polygon(
        polygon,
        w,
        h,
        margin=max(12, min(h, w) // 100),
    )

    sprite = rgba_from_masked_crop(
        image_bgr,
        mask,
        (x1, y1, x2, y2),
    )

    points_px = {
        name: normalized_point_to_px(
            coords,
            w,
            h,
        )
        for name, coords in joints.items()
        if isinstance(coords, (list, tuple)) and len(coords) >= 2
    }

    local_joints = {
        name: (
            coords[0] - x1,
            coords[1] - y1,
        )
        for name, coords in points_px.items()
    }

    # Determine rotation center fallback
    pivot = local_joints.get("proximal") or local_joints.get("root")
    if not pivot and local_joints:
        pivot = list(local_joints.values())[0]
    elif not pivot:
        pivot = (sprite.shape[1] // 2, 0)

    return {
        "name": leg.get("name", "leg"),
        "side": leg.get("side", leg.get("name", "leg")),
        "sprite": sprite,
        "bbox": (x1, y1, x2, y2),
        "joints": local_joints,
        "pivot": pivot,
        "global_joints": points_px,
    }


def get_leg_parts(scene_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    result = []

    for part in scene_data.get("parts", []):
        if not isinstance(part, dict):
            continue

        part_type = str(part.get("type", "")).lower()
        name = str(part.get("name", "")).lower()

        if part_type == "leg" or "leg" in name:
            result.append(part)

    return result


# ============================================================
# WALK CYCLE
# ============================================================

def leg_phase_for_side(side: str) -> float:
    side = side.lower()

    if "front_left" in side:
        return 0.0
    if "back_right" in side:
        return 0.0

    if "front_right" in side:
        return math.pi
    if "back_left" in side:
        return math.pi

    return 0.0


def leg_swing(
    side: str,
    cycle_t: float,
    stride: float,
) -> float:

    phase = leg_phase_for_side(side)
    s = math.sin(
        2.0 * math.pi * cycle_t + phase
    )
    smooth = s * (0.75 + 0.25 * abs(s))
    return smooth * stride * 100.0


def transform_leg_from_joint(
    leg: Dict[str, Any],
    angle: float,
) -> Tuple[np.ndarray, int, int]:

    sprite = leg["sprite"]
    pivot = leg["pivot"]

    rotated, M = rotate_rgba(
        sprite,
        angle,
        center=pivot,
    )

    px, py = pivot
    new_px = M[0, 0] * px + M[0, 1] * py + M[0, 2]
    new_py = M[1, 0] * px + M[1, 1] * py + M[1, 2]

    global_x, global_y = leg["bbox"][0], leg["bbox"][1]

    tx = int(round(global_x + px - new_px))
    ty = int(round(global_y + py - new_py))

    return rotated, tx, ty


def erase_original_leg_from_background(
    base_bgr: np.ndarray,
    leg: Dict[str, Any],
) -> np.ndarray:

    mask = polygon_mask(
        base_bgr.shape[:2],
        leg.get("polygon", []),
        dilation=max(4, min(base_bgr.shape[:2]) // 120),
    )

    if cv2.countNonZero(mask) == 0:
        return base_bgr

    try:
        return cv2.inpaint(
            base_bgr,
            mask,
            5,
            cv2.INPAINT_TELEA,
        )
    except Exception:
        return base_bgr


def make_four_keyframes(
    image_bgr: np.ndarray,
    prepared_legs: List[Dict[str, Any]],
    scene_data: Dict[str, Any],
    stride: float,
    bob_amount: float,
) -> List[np.ndarray]:

    if not prepared_legs:
        return [image_bgr.copy() for _ in range(4)]

    clean_plate = image_bgr.copy()

    for leg in scene_data.get("parts", []):
        if not isinstance(leg, dict):
            continue

        name = str(leg.get("name", "")).lower()
        if "leg" in name or str(leg.get("type", "")).lower() == "leg":
            clean_plate = erase_original_leg_from_background(
                clean_plate,
                leg,
            )

    frames = []
    cycle_positions = [0.00, 0.25, 0.50, 0.75]

    for key_index, cycle_t in enumerate(cycle_positions):
        frame = clean_plate.copy()

        body_y_shift = int(
            math.sin(2.0 * math.pi * cycle_t) *
            image_bgr.shape[0] *
            bob_amount
        )

        for leg in prepared_legs:
            side = leg["side"]
            angle = leg_swing(
                side,
                cycle_t,
                stride,
            )

            if "back" in side.lower():
                angle *= 0.85

            rotated, x, y = transform_leg_from_joint(
                leg,
                angle,
            )

            y += body_y_shift

            frame = overlay_rgba(
                frame,
                rotated,
                x,
                y,
            )

        frames.append(frame)

    return frames


# ============================================================
# SMOOTH INTERPOLATION
# ============================================================

def ease_in_out(t: float) -> float:
    return t * t * (3.0 - 2.0 * t)


def interpolate_keyframe_images(
    keyframes: List[np.ndarray],
    frames_per_segment: int,
) -> List[np.ndarray]:

    if len(keyframes) < 2:
        return keyframes

    result = []

    for i in range(len(keyframes)):
        a = keyframes[i]
        b = keyframes[(i + 1) % len(keyframes)]

        for j in range(frames_per_segment):
            t = j / float(frames_per_segment)
            t = ease_in_out(t)

            blended = cv2.addWeighted(
                a,
                1.0 - t,
                b,
                t,
                0,
            )

            result.append(blended)

    return result


# ============================================================
# WALK-IN / MERGE
# ============================================================

def create_animal_walk_in_frame(
    background: np.ndarray,
    animal_rgba: np.ndarray,
    original_bbox: Tuple[int, int, int, int],
    progress: float,
) -> np.ndarray:

    frame = background.copy()

    x1, y1, x2, y2 = original_bbox
    target_x = x1
    target_y = y1

    ah, aw = animal_rgba.shape[:2]
    start_x = -aw - 20

    current_x = int(
        start_x +
        (target_x - start_x) *
        ease_in_out(progress)
    )

    bob = int(
        math.sin(progress * math.pi * 6.0) *
        background.shape[0] *
        0.008
    )

    frame = overlay_rgba(
        frame,
        animal_rgba,
        current_x,
        target_y + bob,
    )

    return frame


def create_merge_frame(
    animated_frame: np.ndarray,
    original_frame: np.ndarray,
    progress: float,
) -> np.ndarray:

    p = ease_in_out(progress)

    return cv2.addWeighted(
        animated_frame,
        1.0 - p,
        original_frame,
        p,
        0,
    )


# ============================================================
# FULL ANIMATION
# ============================================================

def build_animation(
    image_bgr: np.ndarray,
    animal_rgba: np.ndarray,
    animal_bbox: Tuple[int, int, int, int],
    keyframes: List[np.ndarray],
    total_frames: int,
    walk_cycles: int,
    fps: int,
    mode: str,
    walk_in_fraction: float,
    merge_fraction: float,
) -> List[np.ndarray]:

    original = image_bgr.copy()

    walk_frames = max(
        8,
        int(total_frames * 0.60),
    )

    if walk_cycles > 0:
        walk_frames = max(
            walk_frames,
            walk_cycles * 16,
        )

    smooth_cycle = interpolate_keyframe_images(
        keyframes,
        frames_per_segment=max(2, walk_frames // 4),
    )

    desired_walk_count = max(
        8,
        walk_cycles * 16,
    )

    walk_sequence = []

    for i in range(desired_walk_count):
        walk_sequence.append(
            smooth_cycle[i % len(smooth_cycle)].copy()
        )

    if mode == "Walk in place only":
        return walk_sequence

    walk_in_count = max(
        4,
        int(total_frames * walk_in_fraction),
    )

    walk_in_frames = []
    x1, y1, x2, y2 = animal_bbox
    animal_mask = animal_rgba[:, :, 3]
    background = original.copy()

    full_mask = np.zeros(
        original.shape[:2],
        dtype=np.uint8,
    )
    full_mask[y1:y2, x1:x2] = animal_mask

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (7, 7),
    )
    full_mask = cv2.dilate(full_mask, kernel)

    try:
        background = cv2.inpaint(
            background,
            full_mask,
            7,
            cv2.INPAINT_TELEA,
        )
    except Exception:
        pass

    for i in range(walk_in_count):
        p = i / max(1, walk_in_count - 1)
        walk_in_frames.append(
            create_animal_walk_in_frame(
                background,
                animal_rgba,
                animal_bbox,
                p,
            )
        )

    remaining = max(
        1,
        total_frames - walk_in_count,
    )

    if mode == "Walk in → walk in place":
        merge_count = 0
    else:
        merge_count = max(
            4,
            int(total_frames * merge_fraction),
        )

    cycle_count = max(
        1,
        remaining - merge_count,
    )

    cycle_frames = [
        walk_sequence[i % len(walk_sequence)].copy()
        for i in range(cycle_count)
    ]

    result = walk_in_frames + cycle_frames

    if merge_count > 0:
        animated_last = (
            result[-1]
            if result
            else original.copy()
        )

        for i in range(merge_count):
            p = i / max(1, merge_count - 1)
            result.append(
                create_merge_frame(
                    animated_last,
                    original,
                    p,
                )
            )

    if len(result) > total_frames:
        result = result[:total_frames]

    while len(result) < total_frames:
        result.append(
            result[-1].copy()
            if result
            else original.copy()
        )

    return result


# ============================================================
# EXPORT
# ============================================================

def create_gif(
    frames: List[np.ndarray],
    fps: int,
) -> bytes:

    if not frames:
        return b""

    pil_frames = [
        Image.fromarray(
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        )
        for frame in frames
    ]

    buffer = io.BytesIO()
    duration = max(
        20,
        int(1000 / max(1, fps)),
    )

    pil_frames[0].save(
        buffer,
        format="GIF",
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration,
        loop=0,
        optimize=False,
    )

    return buffer.getvalue()


def create_mp4(
    frames: List[np.ndarray],
    fps: int,
) -> Optional[bytes]:

    if not frames:
        return None

    h, w = frames[0].shape[:2]
    buffer = io.BytesIO()
    temp_path = "/tmp/animal_walk.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(
        temp_path,
        fourcc,
        float(fps),
        (w, h),
    )

    if not writer.isOpened():
        return None

    for frame in frames:
        writer.write(frame)

    writer.release()

    try:
        with open(temp_path, "rb") as f:
            return f.read()
    except Exception:
        return None


def create_frame_zip(
    frames: List[np.ndarray],
) -> bytes:

    buffer = io.BytesIO()
    with zipfile.ZipFile(
        buffer,
        "w",
        compression=zipfile.ZIP_DEFLATED,
    ) as zf:
        for i, frame in enumerate(frames, 1):
            ok, encoded = cv2.imencode(
                ".png",
                frame,
            )
            if ok:
                zf.writestr(
                    f"frame_{i:03d}.png",
                    encoded.tobytes(),
                )
    return buffer.getvalue()


def create_keyframe_zip(
    keyframes: List[np.ndarray],
) -> bytes:

    buffer = io.BytesIO()
    with zipfile.ZipFile(
        buffer,
        "w",
        compression=zipfile.ZIP_DEFLATED,
    ) as zf:
        for i, frame in enumerate(keyframes, 1):
            ok, encoded = cv2.imencode(
                ".png",
                frame,
            )
            if ok:
                zf.writestr(
                    f"walk_keyframe_{i}.png",
                    encoded.tobytes(),
                )
    return buffer.getvalue()


# ============================================================
# VISUALIZE GEMINI DETECTION
# ============================================================

def draw_detection_overlay(
    image_bgr: np.ndarray,
    scene_data: Dict[str, Any],
) -> np.ndarray:

    output = image_bgr.copy()
    h, w = output.shape[:2]

    animal_polygon = scene_data.get("animal_polygon") or []
    pts = normalized_polygon_to_px(
        animal_polygon,
        w,
        h,
    )

    if len(pts) >= 3:
        cv2.polylines(
            output,
            [pts],
            True,
            (0, 180, 0),
            max(2, min(h, w) // 300),
        )

    for part in scene_data.get("parts", []):
        if not isinstance(part, dict):
            continue

        polygon = part.get("polygon") or []
        p = normalized_polygon_to_px(
            polygon,
            w,
            h,
        )

        if len(p) >= 3:
            cv2.polylines(
                output,
                [p],
                True,
                (255, 120, 0),
                max(1, min(h, w) // 500),
            )

        # Draw centerline if available
        centerline = part.get("centerline") or []
        cl_pts = normalized_polygon_to_px(centerline, w, h)
        if len(cl_pts) >= 2:
            cv2.polylines(output, [cl_pts], False, (0, 255, 255), 2)

        joints = part.get("joints") or {}
        for joint_name, joint in joints.items():
            if isinstance(joint, (list, tuple)) and len(joint) >= 2:
                x, y = normalized_point_to_px(
                    joint,
                    w,
                    h,
                )
                cv2.circle(
                    output,
                    (x, y),
                    max(3, min(h, w) // 120),
                    (0, 0, 255),
                    -1,
                )
                cv2.putText(
                    output,
                    str(joint_name),
                    (x + 5, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    max(0.35, min(h, w) / 2500),
                    (0, 0, 255),
                    1,
                    cv2.LINE_AA,
                )

    return output


# ============================================================
# UI
# ============================================================

uploaded_file = st.file_uploader(
    "Upload your hand-drawn scene",
    type=["png", "jpg", "jpeg"],
)

if not uploaded_file:
    st.info(
        "Upload the drawing containing the animal and background. "
        "Then Gemini will identify the animal and its joints."
    )
    st.stop()

file_bytes = uploaded_file.getvalue()

try:
    uploaded_pil = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    png_buffer = io.BytesIO()
    uploaded_pil.save(png_buffer, format="PNG")
    gemini_image_bytes = png_buffer.getvalue()
except Exception as exc:
    st.error(f"Could not prepare the uploaded image for Gemini: {exc}")
    st.stop()

image_np = cv2.imdecode(
    np.frombuffer(file_bytes, np.uint8),
    cv2.IMREAD_COLOR,
)

if image_np is None:
    st.error("Could not read the uploaded image.")
    st.stop()

st.subheader("Original drawing")

c1, c2 = st.columns(2)

with c1:
    st.image(
        cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB),
        caption="Original",
        use_container_width=True,
    )

with c2:
    st.markdown(
        """
### Workflow

**1. Gemini detects the animal**

↓  

**2. Gemini maps its anatomy and joints**

↓  

**3. Python extracts the ORIGINAL animal parts**

↓  

**4. Python creates 4 walking poses**

↓  

**5. The 4 poses are interpolated**

↓  

**6. GIF / MP4 / individual frames are produced**
"""
    )


# ============================================================
# STEP 1
# ============================================================

st.markdown("---")
st.header("🔍 Step 1 — Detect animal and anatomy")

if st.button(
    "Analyze drawing with Gemini",
    type="primary",
    use_container_width=True,
):

    with st.spinner(
        "Gemini is locating the animal, legs and joints..."
    ):

        scene_data = analyze_and_segment_scene(
            gemini_image_bytes,
            GEMINI_API_KEY,
            None,
        )

    if scene_data:
        st.session_state["scene_data"] = scene_data
        st.session_state.pop("animation_frames", None)
        st.session_state.pop("keyframes", None)


# ============================================================
# SHOW ANALYSIS
# ============================================================

if "scene_data" not in st.session_state:
    st.stop()

scene_data = st.session_state["scene_data"]

animal_name = scene_data.get(
    "identified_character",
    "character",
)

st.success(
    f"Detected character: **{animal_name}**"
)

parts = scene_data.get("parts", [])
leg_parts = get_leg_parts(scene_data)

st.write(
    f"Gemini detected **{len(parts)} anatomical parts**, "
    f"including **{len(leg_parts)} leg(s)**."
)

with st.expander("View Gemini analysis JSON"):
    st.json(scene_data)

overlay = draw_detection_overlay(
    image_np,
    scene_data,
)

st.image(
    cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB),
    caption="Gemini detection — green = animal, orange = parts, red = joints, yellow = centerline",
    use_container_width=True,
)


# ============================================================
# STEP 2 — PREPARE LEGS
# ============================================================

st.markdown("---")
st.header("🦵 Step 2 — Prepare walking anatomy")

prepared_legs = []

for leg in leg_parts:
    prepared = prepare_leg_sprite(
        image_np,
        leg,
    )

    if prepared is not None:
        prepared_legs.append(prepared)

if not prepared_legs:
    st.warning(
        "Gemini did not return usable leg polygons and joints. "
        "Try analyzing again with a clearer image."
    )
    st.stop()

st.success(
    f"Prepared {len(prepared_legs)} movable leg(s) from the original drawing."
)

for leg in prepared_legs:
    with st.expander(
        f"🦵 {leg['name']}"
    ):
        st.write("Side:", leg["side"])
        st.write("Global joints:", leg["global_joints"])


# ============================================================
# STEP 3 — GENERATE 4 KEYFRAMES
# ============================================================

st.markdown("---")
st.header("🎬 Step 3 — Generate the 4 walking drawings")

st.write(
    "These are not AI redrawings. Each pose is constructed from the "
    "original hand-drawn leg pixels and rotated around the detected joints."
)

if st.button(
    "🦒 Generate 4 Walking Poses",
    type="primary",
    use_container_width=True,
):

    with st.spinner(
        "Building four hand-drawn walking poses..."
    ):

        keyframes = make_four_keyframes(
            image_np,
            prepared_legs,
            scene_data,
            STRIDE,
            BOB_AMOUNT,
        )

        st.session_state["keyframes"] = keyframes
        st.session_state.pop("animation_frames", None)


if "keyframes" in st.session_state:

    keyframes = st.session_state["keyframes"]
    cols = st.columns(4)

    for i, frame in enumerate(keyframes):
        with cols[i]:
            st.image(
                cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB,
                ),
                caption=f"Walking pose {i + 1}",
                use_container_width=True,
            )

    keyframe_zip = create_keyframe_zip(
        keyframes
    )

    st.download_button(
        "⬇️ Download the 4 separate walking drawings",
        data=keyframe_zip,
        file_name="animal_walk_keyframes.zip",
        mime="application/zip",
        use_container_width=True,
    )


# ============================================================
# STEP 4 — FULL ANIMATION
# ============================================================

if "keyframes" not in st.session_state:
    st.stop()

st.markdown("---")
st.header("🎞️ Step 4 — Render full animation")

st.write(
    f"Mode: **{ANIMATION_MODE}** |  "
    f"{TOTAL_FRAMES} frames  |  "
    f"{FPS} FPS  |  "
    f"{WALK_CYCLES} walking cycle(s)"
)

if st.button(
    "🚀 Render Animation",
    type="primary",
    use_container_width=True,
):

    with st.spinner(
        "Rendering the walking animation..."
    ):

        animal_rgba, animal_bbox, _ = extract_animal(
            image_np,
            scene_data,
        )

        frames = build_animation(
            image_np,
            animal_rgba,
            animal_bbox,
            st.session_state["keyframes"],
            TOTAL_FRAMES,
            WALK_CYCLES,
            FPS,
            ANIMATION_MODE,
            WALK_IN_FRACTION,
            MERGE_FRACTION,
        )

        st.session_state["animation_frames"] = frames

    st.success(
        f"Finished rendering {len(frames)} frames."
    )


# ============================================================
# RESULTS
# ============================================================

if "animation_frames" in st.session_state:

    frames = st.session_state["animation_frames"]

    st.markdown("---")
    st.header("🎉 Result")

    gif_bytes = create_gif(
        frames,
        FPS,
    )

    st.image(
        gif_bytes,
        caption="Generated hand-drawn walking animation",
        use_container_width=True,
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.download_button(
            "⬇️ Download GIF",
            data=gif_bytes,
            file_name="hand_drawn_animal_walk.gif",
            mime="image/gif",
            use_container_width=True,
        )

    with col2:
        mp4_bytes = create_mp4(
            frames,
            FPS,
        )

        if mp4_bytes:
            st.download_button(
                "⬇️ Download MP4",
                data=mp4_bytes,
                file_name="hand_drawn_animal_walk.mp4",
                mime="video/mp4",
                use_container_width=True,
            )
        else:
            st.info(
                "MP4 encoding is unavailable in this environment."
            )

    with col3:
        frame_zip = create_frame_zip(
            frames,
        )

        st.download_button(
            "⬇️ Download all PNG frames",
            data=frame_zip,
            file_name="hand_drawn_animal_animation_frames.zip",
            mime="application/zip",
            use_container_width=True,
        )

    st.markdown("### Individual animation frames")

    preview_count = min(
        12,
        len(frames),
    )

    preview_indices = np.linspace(
        0,
        len(frames) - 1,
        preview_count,
        dtype=int,
    )

    preview_cols = st.columns(4)

    for n, idx in enumerate(preview_indices):
        with preview_cols[n % 4]:
            st.image(
                cv2.cvtColor(
                    frames[idx],
                    cv2.COLOR_BGR2RGB,
                ),
                caption=f"Frame {idx + 1}",
                use_container_width=True,
            )

st.markdown("---")
st.caption(
    "The animation is generated algorithmically from the uploaded "
    "drawing. Gemini is used for visual understanding/geometry, not "
    "for generating replacement artwork."
)

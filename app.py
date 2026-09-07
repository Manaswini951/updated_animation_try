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
    "4-pose walking cycle from the ORIGINAL drawing with auto-model discovery."
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

# Initial default options; can be dynamically updated if client connects successfully
MODEL = st.sidebar.selectbox(
    "Gemini model preference",
    [
        "gemini-2.5-flash",
        "gemini-2.0-flash",
        "gemini-1.5-flash",
        "gemini-2.5-pro",
        "gemini-2.0-pro",
    ],
    index=0,
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
# GEMINI ANALYSIS & MODEL DISCOVERY
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


def get_available_models(client: genai.Client) -> List[str]:
    """Dynamically query the Gemini API to see which models support generateContent."""
    supported_models = []
    try:
        for m in client.models.list():
            # Check if model supports content generation and is active
            model_name = m.name.replace("models/", "")
            if "flash" in model_name or "pro" in model_name:
                supported_models.append(model_name)
    except Exception:
        pass
    
    # Fallback default prioritized list if API listing fails or returns empty
    defaults = ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash", "gemini-2.5-pro"]
    for d in defaults:
        if d not in supported_models:
            supported_models.append(d)
            
    return supported_models


def analyze_and_segment_scene(
    image_bytes: bytes,
    api_key: str,
    preferred_model: str,
) -> Optional[Dict[str, Any]]:

    if not api_key:
        st.error("Please enter a Gemini API key.")
        return None

    try:
        client = genai.Client(api_key=api_key)
    except Exception as e:
        st.error(f"Failed to initialize Gemini client: {e}")
        return None

    # Automatically assemble a robust cascade of models to try
    discovered_models = get_available_models(client)
    
    models_to_try = [preferred_model]
    for m in discovered_models:
        if m not in models_to_try:
            models_to_try.append(m)

    prompt = r"""
You are analyzing a SINGLE hand-drawn animal scene.

The goal is NOT to redraw the animal.
The goal is to give a Python animation program accurate geometry so it
can cut pieces from the ORIGINAL drawing and move them.

Identify the main animal/character and separate it from the background.

IMPORTANT:
- Preserve the artist's original drawing.
- Do NOT invent missing anatomy.
- Do NOT create new artwork.
- Estimate geometry from visible pixels.
- The animal can be a giraffe, horse, cow, dog, cat, rabbit, elephant,
  dinosaur, person, bird, etc.
- If the animal has four legs, identify all four separately.
- For each leg, provide 3 joints along the visible leg:
  proximal_joint, middle_joint, distal_joint.
  For a quadruped these correspond approximately to hip/shoulder,
  knee/elbow, and ankle/wrist/hoof, depending on the leg.
- Coordinates are normalized 0..100 as [y, x].
- Polygons should tightly follow the visible part.
- The animal polygon should tightly surround the WHOLE animal.
- Include enough margin around every polygon to avoid cutting off
  hand-drawn strokes.
- If a tail/head/neck is clearly visible, identify it too.
- Do not make the animal bbox equal to the whole image unless the
  animal really occupies the whole image.

The four walking poses will be created by Python using these joints.
Return ONLY valid JSON.

Required structure:

{
  "identified_character": "giraffe",
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

For a four-legged animal, use these exact leg names when possible:
front_left_leg
front_right_leg
back_left_leg
back_right_leg

If only two legs are visible, return the visible legs and do not invent
the hidden ones.

If there is no animal, return:
{
  "identified_character": "none",
  "animal_bbox": null,
  "animal_polygon": [],
  "parts": [],
  "notes": "No clear animal detected"
}
"""

    last_error = None

    for current_model in models_to_try:
        try:
            response = client.models.generate_content(
                model=current_model,
                contents=[
                    types.Part.from_bytes(
                        data=image_bytes,
                        mime_type="image/png",
                    ),
                    prompt,
                ],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.1,
                ),
            )

            text = clean_json_text(response.text)
            data = json.loads(text)

            if isinstance(data, dict):
                st.toast(f"Successfully connected using model: `{current_model}`", icon="✅")
                return data

        except Exception as exc:
            last_error = exc
            continue

    st.error(f"Gemini analysis failed across all attempted models. Last error: {last_error}")
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
        angle

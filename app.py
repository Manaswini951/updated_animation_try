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

st.title("🦒 Hand-Drawn Animal Walk Animator — Seamless Connected Walk v5")

st.caption(
    "Gemini identifies the animal, anatomy, and custom gait profile. "
    "Python physically animates the ORIGINAL drawing with seamless joint bridging."
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

st.sidebar.markdown("### 🦵 Articulation & Gait")

STEP_ANGLE = st.sidebar.slider(
    "Leg swing intensity",
    2.0,
    25.0,
    10.0,
    0.5,
)

KNEE_BEND = st.sidebar.slider(
    "Knee hinge flexibility",
    0.0,
    20.0,
    8.0,
    0.5,
)

FOOT_LIFT = st.sidebar.slider(
    "Foot clearance lift",
    0.0,
    0.12,
    0.035,
    0.005,
)

BODY_BOB = st.sidebar.slider(
    "Body vertical bob",
    0.0,
    0.04,
    0.006,
    0.001,
)

GROUND_LOCK = st.sidebar.slider(
    "Ground contact friction",
    0.0,
    1.0,
    0.85,
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
    except Exception

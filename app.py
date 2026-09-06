import io
import math
import os

import cv2
import numpy as np
import streamlit as st
from PIL import Image


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Complete Unified Hand-Drawn Animator",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🎨 Complete Unified Hand-Drawn Animator")

st.markdown(
    """
**Sequential Pipeline Executed Per Image:**
1. **Walk-In & Merge:** Character walks in, settles into place, and cross-fades into the full scene with all scenery.
2. **Mandatory In-Scene Color Animation:** The selected color element immediately wiggles, bounces, or glows as part of the image.
3. **Dual Export:** Generates both a **Full Scene GIF** and a **Transparent Overlay GIF** for video editing!
"""
)

MAX_IMAGE_SIZE = 1000

COLOR_ANIMATION_MODES = [
    "Seamless Wiggle & Sway",
    "Rhythmic Bounce & Stretch",
    "Glowing Zoom In/Out",
    "Storytelling Speech Cadence",
]


# ============================================================
# UTILITIES & COLOR EXTRACTION
# ============================================================

def auto_rotate_vertical(image):
    h, w = image.shape[:2]
    if w > h:
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return image


def resize_image(image, max_size=MAX_IMAGE_SIZE):
    h, w = image.shape[:2]
    if max(h, w) <= max_size:
        return image.copy()
    scale = max_size / float(max(h, w))
    return cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def extract_paper_background(image):
    h, w = image.shape[:2]
    border_pixels = np.concatenate([
        image[:15, :].reshape(-1, 3),
        image[-15:, :].reshape(-1, 3),
        image[:, :15].reshape(-1, 3),
        image[:, -15:].reshape(-1, 3)
    ], axis=0)
    bg_color = np.median(border_pixels, axis=0).astype(np.uint8)
    return np.full_like(image, bg_color)


def get_color_name(rgb):
    r, g, b = [int(x) for x in rgb]
    pixel = np.uint8([[[b, g, r]]])
    hsv = cv2.cvtColor(pixel, cv2.COLOR_BGR2HSV)[0][0]
    hue, sat = int(hsv[0]), int(hsv[1])

    if sat < 30:
        return "Neutral/White/Gray"
    if hue < 10 or hue >= 170:
        return "Red"
    elif hue < 25:
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


def extract_dominant_colors(image, max_colors=6):
    small = cv2.resize(image, (150, 150), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    
    valid_pixels = small[hsv[:, :, 1] > 35].reshape(-1, 3)
    if len(valid_pixels) < 100:
        valid_pixels = small.reshape(-1, 3)

    pixels = valid_pixels.astype(np.float32)
    k = min(max_colors, max(2, len(pixels) // 100))
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 5, cv2.KMEANS_PP_CENTERS)
    
    counts = np.bincount(labels.flatten())
    total = sum(counts)

    detected = []
    for idx, center in enumerate(centers):
        b, g, r = [int(x) for x in center]
        hex_code = f"#{r:02x}{g:02x}{b:02x}"
        label = f"{get_color_name((r, g, b))} ({hex_code}) — {counts[idx]/total*100:.1f}%"
        detected.append({
            "label": label,
            "rgb": (r, g, b),
            "bgr": (b, g, r),
            "hex": hex_code,
            "coverage": counts[idx] / total
        })

    detected.sort(key=lambda x: x["coverage"], reverse=True)
    return detected


def make_color_mask(image, target_bgr, tolerance=45):
    diff = np.abs(image.astype(np.int16) - np.array(target_bgr, dtype=np.int16))
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    
    mask = (dist < tolerance).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.GaussianBlur(mask, (3, 3), 0)
    return mask


def extract_character_interactive(image, bbox_pct):
    h, w = image.shape[:2]
    xmin = int((bbox_pct[0] / 100.0) * w)
    ymin = int((bbox_pct[1] / 100.0) * h)
    xmax = int((bbox_pct[2] / 100.0) * w)
    ymax = int((bbox_pct[3] / 100.0) * h)
    
    rect = (xmin, ymin, max(10, xmax - xmin), max(10, ymax - ymin))
    
    gc_mask = np.zeros((h, w), np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    
    try:
        cv2.grabCut(image, gc_mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
        char_mask = np.where((gc_mask == 2) | (gc_mask == 0), 0, 255).astype(np.uint8)
    except Exception:
        char_mask = np.zeros((h, w), np.uint8)
        char_mask[ymin:ymax, xmin:xmax] = 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    char_mask = cv2.morphologyEx(char_mask, cv2.MORPH_CLOSE, kernel)
    char_mask = cv2.GaussianBlur(char_mask, (3, 3), 0)

    ys, xs = np.where(char_mask > 20)
    if len(xs) == 0:
        return None, None, None

    x1, y1 = max(0, np.min(xs) - 5), max(0, np.min(ys) - 5)
    x2, y2 = min(w, np.max(xs) + 5), min(h, np.max(ys) + 5)

    char_crop = image[y1:y2, x1:x2].copy()
    alpha_crop = char_mask[y1:y2, x1:x2].copy()

    return char_crop, alpha_crop, ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


# ============================================================
# LAYER TRANSFORMATIONS & RENDERING ENGINE
# ============================================================

def transform_layer(crop, alpha, scale_x, scale_y, angle, pivot):
    h, w = crop.shape[:2]
    new_w, new_h = max(2, int(w * max(0.05, float(scale_x)))), max(2, int(h * max(0.05, float(scale_y))))

    resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    resized_alpha = cv2.resize(alpha, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    px, py = (pivot[0] / float(w)) * new_w, (pivot[1] / float(h)) * new_h

    M = cv2.getRotationMatrix2D((px, py), angle, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    bw, bh = max(2, int(new_h * sin + new_w * cos)), max(2, int(new_h * cos + new_w * sin))

    M[0, 2] += bw / 2 - px
    M[1, 2] += bh / 2 - py

    warped_c = cv2.warpAffine(resized, M, (bw, bh), borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    warped_a = cv2.warpAffine(resized

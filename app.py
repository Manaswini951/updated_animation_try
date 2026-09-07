import io
import json
import math
import cv2
import numpy as np
import streamlit as st
from PIL import Image
from google import genai
from google.genai import types

# Initialize Streamlit Page Config
st.set_page_config(
    page_title="AI Scene Growth & Character Animator",
    page_icon="🎬",
    layout="wide",
)

st.title("🌱 AI-Powered Scene Growth & Character Animator")
st.markdown(
    """
**Pipeline Workflow:**
1. **AI Segmentation:** Gemini analyzes your drawing, isolates elements (grass, trees, leaves, bunny), and removes backgrounds.
2. **Sequential Scene Growth:** Animate background elements (grass and trees growing upward/scaling in).
3. **Character Walk-In:** The main character walks onto the finished scene.
"""
)

GEMINI_API_KEY = st.sidebar.text_input("Gemini API Key", type="password")

# ============================================================
# STAGE 1: AI OBJECT & LAYER EXTRACTION (GEMINI VISION)
# ============================================================

def analyze_and_segment_scene(image_bytes, api_key):
    """
    Sends the drawing to Gemini to detect individual components 
    and provide layering/positioning metadata.
    """
    if not api_key:
        st.error("Please enter a valid Gemini API Key.")
        return None

    client = genai.Client(api_key=api_key)
    
    prompt = """
    Analyze this hand-drawn scene. Identify all individual background and foreground elements 
    (e.g., grass tufts, trees, bushes, and the main character object).
    Return a JSON object detailing each object, its type ('background_element' or 'character'), 
    its suggested appearance order (1 for background grass/trees, 2 for leaves, 3 for main character), 
    and its bounding box coordinates normalized from 0 to 100 [ymin, xmin, ymax, xmax].
    
    Return ONLY valid JSON in this exact format:
    {
      "scene_elements": [
        {
          "name": "background_tree",
          "type": "background_element",
          "growth_order": 1,
          "box_2d": [ymin, xmin, ymax, xmax]
        },
        {
          "name": "main_bunny",
          "type": "character",
          "growth_order": 3,
          "box_2d": [ymin, xmin, ymax, xmax]
        }
      ]
    }
    """

    try:
        response = client.models.generate_content(
            model='gemini-3.7-flash',
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
                prompt
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json"
            )
        )
        return json.loads(response.text)
    except Exception as e:
        st.error(f"Gemini Scene Analysis Failed: {e}")
        return None

# ============================================================
# STAGE 2: PROCEDURAL COMPOSITION & GROWTH ENGINE
# ============================================================

def extract_sprite_crop(image_np, box):
    """Extracts a sub-image based on normalized coordinates [ymin, xmin, ymax, xmax]."""
    h, w = image_np.shape[:2]
    ymin, xmin, ymax, xmax = box
    
    y1 = int(ymin * h / 100.0)
    y2 = int(ymax * h / 100.0)
    x1 = int(xmin * w / 100.0)
    x2 = int(xmax * w / 100.0)
    
    y1, y2 = max(0, y1), min(h, y2)
    x1, x2 = max(0, x1), min(w, x2)
    
    if y2 <= y1 or x2 <= x1:
        return None, (0, 0)
        
    crop = image_np[y1:y2, x1:x2].copy()
    
    # Remove paper white background to make it transparent/overlay-ready
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, alpha = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY_INV)
    
    # Fallback if thresholding clears everything
    if np.count_nonzero(alpha) < 10:
        alpha = np.full(gray.shape, 255, dtype=np.uint8)
        
    rgba = cv2.cvtColor(crop, cv2.COLOR_BGR2BGRA)
    rgba[:, :, 3] = alpha
    return rgba, (x1, y1)

def render_growth_frame(base_canvas, sprites_data, global_progress):
    """
    Renders elements appearing sequentially: 
    Grass/Trees grow first, followed by leaves, then the character walks in.
    """
    h, w = base_canvas.shape[:2]
    frame = base_canvas.copy()
    if frame.shape[2] == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)

    # Sort sprites by growth order
    sorted_sprites = sorted(sprites_data, key=lambda x: x['growth_order'])

    for item in sorted_sprites:
        order = item['growth_order']
        # Calculate individual appearance threshold based on global progress (0 to 1)
        # Order 1 triggers from 0.0 - 0.4, Order 2 from 0.3 - 0.7, Order 3 (character) from 0.6 - 1.0
        start_trigger = (order - 1) * 0.3
        end_trigger = start_trigger + 0.5
        
        if global_progress < start_trigger:
            continue  # Element hasn't started growing yet
            
        element_progress = min(1.0, (global_progress - start_trigger) / max(1e-6, (end_trigger - start_trigger)))
        
        sprite = item['sprite']
        orig_pos = item['position']
        if sprite is None:
            continue

        sh, sw = sprite.shape[:2]

        if item['type'] == 'background_element':
            # Growth effect: scale up from bottom-center
            current_scale = element_progress
            if current_scale <= 0:
                continue
            new_w = max(2, int(sw * current_scale))
            new_h = max(2, int(sh * current_scale))
            
            resized = cv2.resize(sprite, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            
            # Paste onto canvas anchored at bottom-left of original bounding box
            anchor_x = orig_pos[0]
            anchor_y = orig_pos[1] + sh - new_h  # Grow upward from base
            
            frame = paste_rgba(frame, resized, anchor_x, anchor_y)

        elif item['type'] == 'character':
            # Walk-in effect: character slides in from off-screen left to final position
            walk_progress = element_progress
            start_x = -sw
            target_x = orig_pos[0]
            current_x = int(start_x + (target_x - start_x) * walk_progress)
            
            # Add subtle vertical bobbing while walking
            bob = int(math.sin(walk_progress * math.pi * 6) * 5) if walk_progress < 1.0 else 0
            current_y = orig_pos[1] + bob
            
            frame = paste_rgba(frame, sprite, current_x, current_y)

    return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

def paste_rgba(canvas, sprite, x, y):
    """Helper to cleanly paste a transparent RGBA sprite onto an RGBA canvas."""
    ch, cw = sprite.shape[:2]
    x2, y2 = x + cw, y + ch
    
    if x2 <= 0 or y2 <= 0 or x >= canvas.shape[1] or y >= canvas.shape[0]:
        return canvas

    cx1, cy1 = max(0, x), max(0, y)
    cx2, cy2 = min(canvas.shape[1], x2), min(canvas.shape[0], y2)

    sx1, sy1 = cx1 - x, cy1 - y
    sx2, sy2 = sx1 + (cx2 - cx1), sy1 + (cy2 - cy1)

    s_crop = sprite[sy1:sy2, sx1:sx2]
    a = (s_crop[:, :, 3].astype(np.float32) / 255.0)[:, :, None]
    
    bg_crop = canvas[cy1:cy2, cx1:cx2].astype(np.float32)

    blended = s_crop.astype(np.float32) * a + bg_crop * (1.0 - a)
    canvas[cy1:cy2, cx1:cx2] = np.clip(blended, 0, 255).astype(np.uint8)
    return canvas

def create_gif(frames, fps=12):
    buffer = io.BytesIO()
    pil_frames = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames]
    duration = int(1000 / fps)
    pil_frames[0].save(
        buffer, format="GIF", save_all=True, append_images=pil_frames[1:],
        duration=duration, loop=0
    )
    return buffer.getvalue()

# ============================================================
# STAGE 3: STREAMLIT USER INTERFACE EXECUTION
# ============================================================

uploaded_file = st.file_uploader("Upload Scene Drawing", type=["png", "jpg", "jpeg"])

if uploaded_file and GEMINI_API_KEY:
    file_bytes = uploaded_file.read()
    image_np = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)
    
    st.image(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB), caption="Original Uploaded Scene", width=500)

    if st.button("🔍 Step 1: Extract Scene Elements with Gemini", type="primary"):
        with st.spinner("Gemini is analyzing and segmenting objects from your drawing..."):
            scene_data = analyze_and_segment_scene(file_bytes, GEMINI_API_KEY)
            
        if scene_data and "scene_elements" in scene_data:
            st.session_state["scene_elements"] = scene_data["scene_elements"]
            st.success(f"Successfully isolated {len(scene_data['scene_elements'])} scene components!")
            st.json(scene_data)

    if "scene_elements" in st.session_state:
        st.markdown("### 🎬 Step 2: Render Growth & Walk-In Animation")
        
        total_frames = st.slider("Animation Frame Count", 12, 60, 30)
        fps = st.slider("Frames Per Second (FPS)", 6, 24, 12)

        if st.button("🚀 Generate Scene Growth Animation"):
            with st.spinner("Assembling and rendering growing background elements and character walk-in..."):
                # Clean white background canvas plate
                white_canvas = np.full(image_np.shape, 255, dtype=np.uint8)
                
                # Pre-extract sprite crops for each element found by Gemini
                processed_sprites = []
                for element in st.session_state["scene_elements"]:
                    sprite, pos = extract_sprite_crop(image_np, element["box_2d"])
                    processed_sprites.append({
                        "name": element["name"],
                        "type": element["type"],
                        "growth_order": element["growth_order"],
                        "sprite": sprite,
                        "position": pos
                    })

                # Render frame sequence
                frames = []
                progress_bar = st.progress(0, text="Rendering growth progression...")
                
                for i in range(total_frames):
                    progress = i / max(1, total_frames - 1)
                    frame = render_growth_frame(white_canvas, processed_sprites, progress)
                    frames.append(frame)
                    progress_bar.progress((i + 1) / total_frames)

                progress_bar.empty()

                # Build final GIF output
                gif_bytes = create_gif(frames, fps=fps)

                st.markdown("### 🎉 Rendered Animation Output")
                st.image(gif_bytes, caption="Growing Scene + Character Walk-In", width=500)
                
                st.download_button(
                    label="⬇️ Download Growth Animation GIF",
                    data=gif_bytes,
                    file_name="scene_growth_animation.gif",
                    mime="image/gif"
                )

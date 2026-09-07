import io
import json
import math
import cv2
import numpy as np
import streamlit as st
from PIL import Image
from google import genai
from google.genai import types

st.set_page_config(
    page_title="AI Dynamic Scene & Character Animator",
    page_icon="🎬",
    layout="wide",
)

st.title("🦒 AI-Powered Dynamic Character & Scene Animator")
st.markdown(
    """
**Pipeline Workflow:**
1. **Dynamic AI Recognition:** Gemini automatically figures out what animal/character you drew (e.g., a giraffe, rabbit, etc.) and maps its unique parts.
2. **Plate Preserved:** Keeps your original marker colors and background textures intact.
3. **Sequential Assembly:** Scenery grows into place first, followed by the character's custom body parts assembling.
"""
)

GEMINI_API_KEY = st.sidebar.text_input("Gemini API Key", type="password")

def analyze_and_segment_scene(image_bytes, api_key):
    if not api_key:
        st.error("Please enter a valid Gemini API Key.")
        return None

    client = genai.Client(api_key=api_key)
    
    prompt = """
    Analyze this hand-drawn scene. 
    1. Identify what animal or character this is (e.g., giraffe, rabbit, cat, dinosaur, etc.).
    2. Identify any background/scenery elements (grass, trees, bushes).
    3. Break down the character into its specific anatomical parts based on what animal it actually is 
       (e.g., if it's a giraffe, extract parts like 'head', 'horns', 'long_neck', 'body', 'leg_1', 'leg_2', 'leg_3', 'leg_4', 'tail').
    
    For each item, specify its type ('background_element' or 'character_part'), 
    its animation/growth order (backgrounds grow first at 1 or 2, character parts assemble last at order 3), 
    and its precise bounding box coordinates normalized from 0 to 100 [ymin, xmin, ymax, xmax].
    
    Return ONLY valid JSON in this exact format:
    {
      "identified_character": "giraffe",
      "scene_elements": [
        {
          "name": "background_element_1",
          "type": "background_element",
          "growth_order": 1,
          "box_2d": [ymin, xmin, ymax, xmax]
        },
        {
          "name": "head",
          "type": "character_part",
          "growth_order": 3,
          "box_2d": [ymin, xmin, ymax, xmax]
        },
        {
          "name": "long_neck",
          "type": "character_part",
          "growth_order": 3,
          "box_2d": [ymin, xmin, ymax, xmax]
        }
      ]
    }
    """

    fallback_models = ['gemini-3.7-flash', 'gemini-3.6-flash', 'gemini-2.5-flash']

    for model_name in fallback_models:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=[
                    types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
                    prompt
                ],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json"
                )
            )
            return json.loads(response.text)
        except Exception:
            continue

    st.error("All available Gemini models are currently busy (503). Please try again shortly.")
    return None

def extract_element_crop(image_np, box):
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
    rgba = cv2.cvtColor(crop, cv2.COLOR_BGR2BGRA)
    return rgba, (x1, y1)

def render_growth_frame(base_plate, sprites_data, global_progress):
    h, w = base_plate.shape[:2]
    frame = cv2.cvtColor(base_plate, cv2.COLOR_BGR2BGRA)
    
    sorted_sprites = sorted(sprites_data, key=lambda x: x['growth_order'])

    for item in sorted_sprites:
        order = item['growth_order']
        start_trigger = (order - 1) * 0.25
        end_trigger = start_trigger + 0.45
        
        sprite = item['sprite']
        orig_pos = item['position']
        if sprite is None:
            continue

        sh, sw = sprite.shape[:2]
        element_progress = (global_progress - start_trigger) / max(1e-6, (end_trigger - start_trigger))
        element_progress = float(np.clip(element_progress, 0.0, 1.0))

        if global_progress < start_trigger:
            frame[orig_pos[1]:orig_pos[1]+sh, orig_pos[0]:orig_pos[0]+sw] = [255, 255, 255, 255]
            continue

        if item['type'] == 'background_element':
            current_scale = element_progress
            if current_scale <= 0:
                frame[orig_pos[1]:orig_pos[1]+sh, orig_pos[0]:orig_pos[0]+sw] = [255, 255, 255, 255]
                continue
                
            new_w = max(2, int(sw * current_scale))
            new_h = max(2, int(sh * current_scale))
            
            resized = cv2.resize(sprite, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            frame[orig_pos[1]:orig_pos[1]+sh, orig_pos[0]:orig_pos[0]+sw] = [255, 255, 255, 255]
            
            anchor_x = orig_pos[0]
            anchor_y = orig_pos[1] + sh - new_h  
            frame = paste_rgba(frame, resized, anchor_x, anchor_y)

        elif item['type'] == 'character_part':
            walk_progress = element_progress
            start_x = -sw - 40
            target_x = orig_pos[0]
            current_x = int(start_x + (target_x - start_x) * walk_progress)
            
            bob = int(math.sin(walk_progress * math.pi * 8) * 3) if walk_progress < 1.0 else 0
            current_y = orig_pos[1] + bob
            
            if walk_progress < 1.0:
                frame[orig_pos[1]:orig_pos[1]+sh, orig_pos[0]:orig_pos[0]+sw] = [255, 255, 255, 255]
                
            frame = paste_rgba(frame, sprite, current_x, current_y)

    return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

def paste_rgba(canvas, sprite, x, y):
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

uploaded_file = st.file_uploader("Upload Drawing (Giraffe, Rabbit, etc.)", type=["png", "jpg", "jpeg"])

if uploaded_file and GEMINI_API_KEY:
    file_bytes = uploaded_file.read()
    image_np = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)
    
    st.image(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB), caption="Original Uploaded Drawing", width=500)

    if st.button("🔍 Step 1: Automatically Detect Animal & Parts with Gemini", type="primary"):
        with st.spinner("Gemini is analyzing the drawing's unique anatomy..."):
            scene_data = analyze_and_segment_scene(file_bytes, GEMINI_API_KEY)
            
        if scene_data and "scene_elements" in scene_data:
            st.session_state["scene_elements"] = scene_data["scene_elements"]
            detected_name = scene_data.get("identified_character", "character")
            st.success(f"Successfully recognized a **{detected_name}** and mapped {len(scene_data['scene_elements'])} parts!")
            st.json(scene_data)

    if "scene_elements" in st.session_state:
        st.markdown("### 🎬 Step 2: Render Custom Animation")
        
        total_frames = st.slider("Animation Frame Count", 15, 60, 30)
        fps = st.slider("Frames Per Second (FPS)", 6, 24, 12)

        if st.button("🚀 Render Dynamic Assembly GIF"):
            with st.spinner("Generating animation sequence..."):
                processed_sprites = []
                for element in st.session_state["scene_elements"]:
                    sprite, pos = extract_element_crop(image_np, element["box_2d"])
                    processed_sprites.append({
                        "name": element["name"],
                        "type": element["type"],
                        "growth_order": element["growth_order"],
                        "sprite": sprite,
                        "position": pos
                    })

                frames = []
                progress_bar = st.progress(0, text="Rendering frames...")
                
                for i in range(total_frames):
                    progress = i / max(1, total_frames - 1)
                    frame = render_growth_frame(image_np, processed_sprites, progress)
                    frames.append(frame)
                    progress_bar.progress((i + 1) / total_frames)

                progress_bar.empty()

                gif_bytes = create_gif(frames, fps=fps)

                st.markdown("### 🎉 Result")
                st.image(gif_bytes, caption="Dynamic Animation Preview", width=500)
                
                st.download_button(
                    label="⬇️ Download Animation GIF",
                    data=gif_bytes,
                    file_name="dynamic_animation.gif",
                    mime="application/gif"
                )

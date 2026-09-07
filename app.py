import io
import json
import math
import cv2
import numpy as np
import streamlit as st
from PIL import Image
from google import genai
from google.genai import types
from skimage.transform import PiecewiseAffineTransform, warp

# Initialize Gemini API
GEMINI_API_KEY = st.sidebar.text_input("Gemini API Key", type="password")

# ============================================================
# PHASE 1: AI JOINT DETECTION (GEMINI VISION)
# ============================================================

def detect_character_skeleton(image_bytes, api_key):
    """
    Sends drawing to Gemini to identify key joints for limb bending.
    """
    if not api_key:
        st.error("Please enter a valid Gemini API Key.")
        return None

    client = genai.Client(api_key=api_key)
    
    prompt = """
    Analyze this drawing. Identify the character and return a JSON object with 2D keypoint coordinates normalized from 0 to 100 for:
    - head
    - neck
    - left_shoulder, left_elbow, left_wrist
    - right_shoulder, right_elbow, right_wrist
    - left_hip, left_knee, left_ankle
    - right_hip, right_knee, right_ankle
    
    Return ONLY valid JSON in this structure:
    {
      "character_detected": true,
      "joints": {
         "head": [x, y],
         "neck": [x, y],
         "left_knee": [x, y]
      }
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
        st.error(f"Gemini Joint Detection Failed: {e}")
        return None

# ============================================================
# PHASE 2: LIMB DEFORMATION & BENDING (MESH WARPING)
# ============================================================

def bend_limb_mesh(image, joint_start, joint_mid, joint_end, bend_angle_deg):
    """
    Uses Piecewise Affine Transformation to deform image mesh around a joint.
    """
    h, w = image.shape[:2]
    
    p_start = np.array([joint_start[0] * w / 100.0, joint_start[1] * h / 100.0])
    p_mid   = np.array([joint_mid[0] * w / 100.0,   joint_mid[1] * h / 100.0])
    p_end   = np.array([joint_end[0] * w / 100.0,   joint_end[1] * h / 100.0])

    angle_rad = math.radians(bend_angle_deg)
    rot_matrix = np.array([
        [math.cos(angle_rad), -math.sin(angle_rad)],
        [math.sin(angle_rad),  math.cos(angle_rad)]
    ])

    p_end_bent = p_mid + np.dot(rot_matrix, (p_end - p_mid))

    src_points = np.array([p_start, p_mid, p_end, [0, 0], [w, 0], [0, h], [w, h]])
    dst_points = np.array([p_start, p_mid, p_end_bent, [0, 0], [w, 0], [0, h], [w, h]])

    tform = PiecewiseAffineTransform()
    tform.estimate(dst_points, src_points)
    
    warped = warp(image, tform, output_shape=(h, w))
    return (warped * 255).astype(np.uint8)

def create_gif_from_frames(frames, fps=12):
    """
    Compiles a list of RGB numpy image frames into a GIF byte stream.
    """
    pil_frames = [Image.fromarray(f) for f in frames]
    buffer = io.BytesIO()
    duration = int(1000 / fps)
    pil_frames[0].save(
        buffer,
        format="GIF",
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration,
        loop=0
    )
    return buffer.getvalue()

# ============================================================
# PHASE 3: STREAMLIT WORKFLOW
# ============================================================

st.title("🤖 AI-Rigged Hand-Drawn Character Animator")

uploaded_file = st.file_uploader("Upload Drawing", type=["png", "jpg", "jpeg"])

if uploaded_file and GEMINI_API_KEY:
    file_bytes = uploaded_file.read()
    image_np = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)
    image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

    st.image(image_rgb, caption="Source Artwork", width=400)

    if st.button("🔍 Step 1: Detect Character & Joints with Gemini"):
        with st.spinner("Analyzing character skeleton via Gemini API..."):
            skeleton_data = detect_character_skeleton(file_bytes, GEMINI_API_KEY)
            
        if skeleton_data and skeleton_data.get("character_detected"):
            st.session_state["skeleton"] = skeleton_data["joints"]
            st.success("Joints identified successfully!")
            st.json(skeleton_data["joints"])

    if "skeleton" in st.session_state:
        st.markdown("### 🦵 Animation Controls")
        joints = st.session_state["skeleton"]

        col1, col2 = st.columns(2)
        with col1:
            max_knee_angle = st.slider("Max Knee Bend Angle", 0, 45, 25)
            max_elbow_angle = st.slider("Max Elbow Bend Angle", 0, 45, 20)
        with col2:
            num_frames = st.slider("Total Animation Frames", 12, 48, 24)
            fps = st.slider("Frames Per Second (FPS)", 6, 24, 12)

        if st.button("🎬 Generate & Render GIF Animation", type="primary"):
            frames = []
            progress_bar = st.progress(0, text="Rendering animation sequence...")

            for i in range(num_frames):
                # Calculate smooth cyclical bending using sine waves
                t = (i / num_frames) * 2 * math.pi
                cur_knee_angle = math.sin(t) * max_knee_angle
                cur_elbow_angle = math.sin(t + math.pi / 2) * max_elbow_angle

                frame = image_rgb.copy()

                # Bend Left Leg
                if "left_hip" in joints and "left_knee" in joints and "left_ankle" in joints:
                    frame = bend_limb_mesh(
                        frame, 
                        joints["left_hip"], 
                        joints["left_knee"], 
                        joints["left_ankle"], 
                        cur_knee_angle
                    )

                # Bend Left Arm
                if "left_shoulder" in joints and "left_elbow" in joints and "left_wrist" in joints:
                    frame = bend_limb_mesh(
                        frame, 
                        joints["left_shoulder"], 
                        joints["left_elbow"], 
                        joints["left_wrist"], 
                        cur_elbow_angle
                    )

                frames.append(frame)
                progress_bar.progress((i + 1) / num_frames)

            progress_bar.empty()

            # Compile into GIF
            gif_data = create_gif_from_frames(frames, fps=fps)

            st.markdown("### 🎉 Rendered Animation Result")
            st.image(gif_data, caption="Animated Hand-Drawn Character", width=400)
            
            st.download_button(
                label="⬇️ Download Animated GIF",
                data=gif_data,
                file_name="character_animation.gif",
                mime="image/gif"
            )

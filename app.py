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
        # Updated model identifier to gemini-3.6-flash
        response = client.models.generate_content(
            model='gemini-3.6-flash',
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
    Uses Piecewise Affine Transformation to deform image mesh around a joint (e.g. Knee/Elbow).
    """
    h, w = image.shape[:2]
    
    # Map normalized coordinates to pixel values
    p_start = np.array([joint_start[0] * w / 100.0, joint_start[1] * h / 100.0])
    p_mid   = np.array([joint_mid[0] * w / 100.0,   joint_mid[1] * h / 100.0])
    p_end   = np.array([joint_end[0] * w / 100.0,   joint_end[1] * h / 100.0])

    # Calculate rotation for the lower part of the limb
    angle_rad = math.radians(bend_angle_deg)
    rot_matrix = np.array([
        [math.cos(angle_rad), -math.sin(angle_rad)],
        [math.sin(angle_rad),  math.cos(angle_rad)]
    ])

    # Rotate end joint relative to mid joint (knee/elbow)
    p_end_bent = p_mid + np.dot(rot_matrix, (p_end - p_mid))

    # Construct Source and Destination Control Points
    src_points = np.array([p_start, p_mid, p_end, [0, 0], [w, 0], [0, h], [w, h]])
    dst_points = np.array([p_start, p_mid, p_end_bent, [0, 0], [w, 0], [0, h], [w, h]])

    # Warp image texture along control mesh
    tform = PiecewiseAffineTransform()
    tform.estimate(dst_points, src_points)
    
    warped = warp(image, tform, output_shape=(h, w))
    return (warped * 255).astype(np.uint8)

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
        st.markdown("### 🦵 Limb Bending Controls")
        joints = st.session_state["skeleton"]

        knee_angle = st.slider("Left Knee Bend Angle", -45, 45, 15)
        elbow_angle = st.slider("Left Elbow Bend Angle", -45, 45, -20)

        if st.button("🎬 Render Pose Bends"):
            # Bend Left Leg (Hip -> Knee -> Ankle)
            frame = image_rgb.copy()
            if "left_hip" in joints and "left_knee" in joints and "left_ankle" in joints:
                frame = bend_limb_mesh(
                    frame, 
                    joints["left_hip"], 
                    joints["left_knee"], 
                    joints["left_ankle"], 
                    knee_angle
                )

            # Bend Left Arm (Shoulder -> Elbow -> Wrist)
            if "left_shoulder" in joints and "left_elbow" in joints and "left_wrist" in joints:
                frame = bend_limb_mesh(
                    frame, 
                    joints["left_shoulder"], 
                    joints["left_elbow"], 
                    joints["left_wrist"], 
                    elbow_angle
                )

            st.image(frame, caption="Deformed Character Pose", width=400)

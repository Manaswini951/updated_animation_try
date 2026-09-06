import cv2
import numpy as np
import streamlit as st

def paste_layer(base_img, layer_img, offset=(0, 0)):
    # Standardize both images to RGBA to avoid channel shape mismatches (3 vs 4 channels)
    if base_img.shape[2] == 3:
        base_img = cv2.cvtColor(base_img, cv2.COLOR_BGR2BGRA)
    if layer_img.shape[2] == 3:
        layer_img = cv2.cvtColor(layer_img, cv2.COLOR_BGR2BGRA)

    h_base, w_base = base_img.shape[:2]
    h_layer, w_layer = layer_img.shape[:2]
    x_offset, y_offset = offset

    # Calculate overlapping regions
    x1, y1 = max(0, x_offset), max(0, y_offset)
    x2, y2 = min(w_base, x_offset + w_layer), min(h_base, y_offset + h_layer)

    if x1 >= x2 or y1 >= y2:
        return base_img

    overlay_x1, overlay_y1 = max(0, -x_offset), max(0, -y_offset)
    overlay_x2 = overlay_x1 + (x2 - x1)
    overlay_y2 = overlay_y1 + (y2 - y1)

    # Extract overlapping regions
    base_crop = base_img[y1:y2, x1:x2].astype(float)
    layer_crop = layer_img[overlay_y1:overlay_y2, overlay_x1:overlay_x2].astype(float)

    # Perform alpha blending safely
    alpha_layer = layer_crop[:, :, 3:4] / 255.0
    alpha_base = base_crop[:, :, 3:4] / 255.0

    out_alpha = alpha_layer + alpha_base * (1.0 - alpha_layer)
    # Prevent division by zero
    safe_out_alpha = np.where(out_alpha == 0, 1.0, out_alpha)

    out_rgb = (
        layer_crop[:, :, :3] * alpha_layer
        + base_crop[:, :, :3] * alpha_base * (1.0 - alpha_layer)
    ) / safe_out_alpha

    blended = np.zeros_like(base_crop)
    blended[:, :, :3] = out_rgb
    blended[:, :, 3:4] = out_alpha * 255.0

    base_img[y1:y2, x1:x2] = blended.astype(np.uint8)
    return base_img


def render_sequential_frame(base_canvas, layers):
    result = base_canvas.copy()
    # Ensure canvas is 4-channel RGBA before layer application
    if result.shape[2] == 3:
        result = cv2.cvtColor(result, cv2.COLOR_BGR2BGRA)

    for layer in layers:
        result = paste_layer(result, layer)
    return result


def compute_distance(diff_array):
    # Clamp negative floating point values to 0 before taking square root
    sum_sq = np.sum(diff_array**2, axis=2)
    return np.sqrt(np.maximum(0, sum_sq))


# Streamlit UI Setup
st.set_page_config(page_title="Layer Processor", layout="wide")

st.title("Image Layer Processor")

uploaded_file = st.sidebar.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)

    # Use width='stretch' instead of deprecated use_container_width=True
    st.image(image, caption="Uploaded Image", width="stretch")

    # Example frame generation trigger
    if st.button("Render Processing", width="stretch"):
        processed_img = render_sequential_frame(image, [image])
        st.image(processed_img, caption="Processed Output", width="stretch")

# app.py
import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from streamlit_drawable_canvas import st_canvas
from PIL import Image, ImageOps

# --------------------------
# Load trained model
# --------------------------
@st.cache_resource
def load_trained_model():
    return load_model("mnist_model.h5")

model = load_trained_model()

# --------------------------
# Preprocessing (same as train.py)
# --------------------------
def preprocess(pil_img):
    img = pil_img.convert("L")
    img = ImageOps.invert(img)
    arr = np.array(img)

    # Threshold to clean noise
    thresh = 30
    arr = (arr > thresh).astype(np.uint8) * 255

    ys, xs = np.where(arr > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    arr = arr[y0:y1, x0:x1]

    h, w = arr.shape
    if h > w:
        new_h, new_w = 20, max(1, int(round(20 * w / h)))
    else:
        new_w, new_h = 20, max(1, int(round(20 * h / w)))
    digit = Image.fromarray(arr).resize((new_w, new_h), Image.BILINEAR)

    canvas = Image.new("L", (28, 28))
    top = (28 - new_h) // 2
    left = (28 - new_w) // 2
    canvas.paste(digit, (left, top))

    out = np.array(canvas).astype("float32") / 255.0
    out = np.expand_dims(out, axis=-1)
    out = np.expand_dims(out, axis=0)
    return out

# --------------------------
# Streamlit UI
# --------------------------
st.title("🖌️ Handwritten Digit Recognition")
st.write("Draw a digit (0–9) or upload an image to predict automatically.")

option = st.radio("Choose Input Mode", ["Draw Digit", "Upload Image"])

# --------------------------
# Drawing Mode
# --------------------------
if option == "Draw Digit":
    st.write("✏️ Draw a digit below:")

    canvas_result = st_canvas(
        fill_color="rgba(255,255,255,1)",
        stroke_width=14,
        stroke_color="black",
        background_color="white",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas",
    )

    if canvas_result.image_data is not None:
        rgba = canvas_result.image_data.astype("uint8")
        img = Image.fromarray(rgba).convert("RGB")
        st.image(img, caption="Your Drawing", width=200)

        # Auto predict when drawing exists
        processed_img = preprocess(img)
        if processed_img is not None:
            pred = model.predict(processed_img, verbose=0)
            st.success(f"Predicted Digit: **{np.argmax(pred)}**")
        else:
            st.warning("⚠️ No strokes detected. Try drawing thicker or larger.")

# --------------------------
# Upload Mode
# --------------------------
elif option == "Upload Image":
    uploaded_file = st.file_uploader("📂 Upload a digit image", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", width=200)

        processed_img = preprocess(image)
        if processed_img is not None:
            pred = model.predict(processed_img, verbose=0)
            st.success(f"Predicted Digit: **{np.argmax(pred)}**")
        else:
            st.warning("⚠️ No digit detected. Please upload a clearer image.")

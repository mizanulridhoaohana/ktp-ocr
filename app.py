import streamlit as st
from inference_sdk import InferenceHTTPClient
from PIL import Image, ImageDraw
import easyocr
import numpy as np
import os

# Initialize EasyOCR
reader = easyocr.Reader(['id'])

@st.cache_data
def process_image(image_path, model_id):
    CLIENT = InferenceHTTPClient(
        api_url="https://detect.roboflow.com",
        api_key="KsbTGBigSIYOwk4226BL"
    )

    # Perform inference
    result = CLIENT.infer(image_path, model_id=model_id)

    # Load the original image
    original_image = Image.open(image_path)
    annotated_image = original_image.copy()
    draw = ImageDraw.Draw(annotated_image)

    # Process each prediction
    for prediction in result['predictions']:
        x1 = int(prediction['x'] - prediction['width'] / 2)
        y1 = int(prediction['y'] - prediction['height'] / 2)
        x2 = int(prediction['x'] + prediction['width'] / 2)
        y2 = int(prediction['y'] + prediction['height'] / 2)

        # Draw bounding box
        draw.rectangle([x1, y1, x2, y2], outline="red")

        # Crop the region around the bounding box
        cropped_image = original_image.crop((x1, y1, x2, y2))

        # Perform OCR using EasyOCR
        ocr_results = reader.readtext(np.array(cropped_image))

        # Print or use OCR result as needed
        for result in ocr_results:
            st.write(f"OCR {prediction.get('class', 'Unknown')}: {result[1]}")

    # Save the annotated image
    annotated_image_path = "ktp_annotated.png"
    annotated_image.save(annotated_image_path)
    st.image(annotated_image, caption="Annotated Image")

# Streamlit UI
st.title("KTP Image Processing")

uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image")

    model_id = "ktp-object-detection/1"  # Adjust as needed

    if st.button("Process Image"):
        with open("temp_image.png", "wb") as f:
            f.write(uploaded_file.read())
       
        process_image("temp_image.png", model_id)
        os.remove("temp_image.png")

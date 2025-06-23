import streamlit as st # type: ignore
import torch # type: ignore
import torchvision.transforms as transforms # type: ignore
import torchvision.models as models # type: ignore
from PIL import Image # type: ignore
import requests

# Load ImageNet class labels
LABELS_URL = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
imagenet_classes = requests.get(LABELS_URL).text.strip().split("\n")

@st.cache_resource
def load_model():
    model = models.efficientnet_b0(pretrained=True)
    model.eval()
    return model

def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    return transform(image).unsqueeze(0)

def classify_image(model, image):
    try:
        img_tensor = preprocess_image(image)
        outputs = model(img_tensor)
        probs = torch.nn.functional.softmax(outputs[0], dim=0)
        top_probs, top_idxs = torch.topk(probs, 3)
        return [(imagenet_classes[i], float(p)) for i, p in zip(top_idxs, top_probs)]
    except Exception as e:
        st.error(f"Classification error: {str(e)}")
        return None

def main():
    st.set_page_config(page_title="Image Classifier", page_icon="📸", layout="centered")
    st.title("🧠 AI Image Classifier")
    st.markdown("Upload an image **or** use your **webcam** to capture a photo and classify it.")

    model = load_model()

    # Upload option
    uploaded_file = st.file_uploader("📤 Upload an image...", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)
        if st.button("Classify Uploaded Image"):
            with st.spinner("Analyzing..."):
                predictions = classify_image(model, image)
                if predictions:
                    st.subheader("📊 Predictions:")
                    for label, prob in predictions:
                        st.write(f"**{label}**: {prob:.2%}")

    st.divider()

    # Webcam capture option with controlled permission prompt
    st.subheader("📷 Capture Photo From Webcam")

    # Initialize session state variable if not present
    if "camera_open" not in st.session_state:
        st.session_state.camera_open = False

    if not st.session_state.camera_open:
        if st.button("Open Camera"):
            st.session_state.camera_open = True

    if st.session_state.camera_open:
        st.info("👆 After capturing, press the button to classify.")
        webcam_image = st.camera_input("")
        if webcam_image:
            image = Image.open(webcam_image).convert("RGB")
            st.image(image, caption="Captured Image", use_container_width=True)
            if st.button("Classify Captured Image"):
                with st.spinner("Analyzing..."):
                    predictions = classify_image(model, image)
                    if predictions:
                        st.subheader("📊 Predictions:")
                        for label, prob in predictions:
                            st.write(f"**{label}**: {prob:.2%}")

if __name__ == "__main__":
    main()



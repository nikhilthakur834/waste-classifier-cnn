import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

# Class names in alphabetical order (same order used during training)
CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

# One building block of the CNN: Conv -> BatchNorm -> ReLU -> MaxPool
def block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1),
        nn.BatchNorm2d(cout),
        nn.ReLU(),
        nn.MaxPool2d(2))

# Same model architecture as in training (needed to load the saved weights)
class WasteCNN(nn.Module):
    def __init__(self, n_classes=6):
        super().__init__()
        self.features = nn.Sequential(block(3, 32), block(32, 64), block(64, 128), block(128, 256))
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Dropout(0.4), nn.Linear(256, n_classes))

    def forward(self, x):
        return self.head(self.features(x))

# Load the model only once and reuse it (makes the app faster)
@st.cache_resource
def load_model():
    model = WasteCNN(len(CLASSES))
    model.load_state_dict(torch.load("best_cnn_v2.pt", map_location="cpu"))
    model.eval()  # evaluation mode: dropout off
    return model

# Same preprocessing as the test set during training: resize + normalize
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

st.title("Waste Classifier")
st.write("Upload a photo of a waste item and the CNN will predict its type.")

file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if file is not None:
    image = Image.open(file).convert("RGB")
    st.image(image, caption="Uploaded image", width=300)

    x = transform(image).unsqueeze(0)  # add batch dimension: (1, 3, 128, 128)
    with torch.no_grad():              # no training, so no gradients needed
        probs = torch.softmax(load_model()(x), dim=1)[0]  # scores -> probabilities

    best = probs.argmax().item()  # index of the class with the highest probability

    # Show a clean result without the probability chart.
    # If the model is not confident, say so instead of giving a hard answer.
    if probs[best].item() < 0.4:
        st.warning(f"Not sure, but it looks like: {CLASSES[best]}")
    else:
        st.success(f"Prediction: {CLASSES[best]}")

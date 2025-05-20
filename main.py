# -*- coding: utf-8 -*-



import os
import json
from PIL import Image

import numpy as np
import tensorflow as tf
import streamlit as st
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms, models
import torchvision.transforms as transforms
from torchvision import transforms, datasets

# Load class index
with open("/workspaces/Trained_PDD/app/class_indices.json", "r") as f:
    class_to_idx = json.load(f)

#  Ensures the values are integers for correct indexing
class_to_idx = {k: str(v) for k, v in class_to_idx.items()}
idx_to_class = {v: k for k, v in class_to_idx.items()}





working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = f"{working_dir}/workspaces/Trained_PDD/app/Final_Trained_Project_Deep_Learning_for_PDD.pth"

# Load the pre-trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load("/workspaces/Trained_PDD/app/Final_Trained_Project_Deep_Learning_for_PDD.pth", map_location=device, weights_only=False)
model.eval()



#image parameters
img_size = 224
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])

                         
])


# Function to Load and Preprocess the Image using PillOW"""
def load_and_preprocess_image(image_path, target_size=(224, 224)):
    # Load the image
    img = Image.open(image_path)
    # Resize the image
    img = img.resize(target_size)
    # Convert the image to a numpy array
    img_array = np.array(img)
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    # Scale the image values to [0, 1]
    img_array = img_array.astype('float32') / 255.
    return img_array

# Streamlit app
st.title("PlantVillage Disease Classifier 🌿")
st.write("Upload a plant to predict the plant disease.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])


if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        
        outputs = model(img_tensor)
        _, predicted_idx = torch.max(outputs, 1)
        
       
        predicted_class = idx_to_class[str(predicted_idx.item())]

    st.success(f"✅ Predicted class: **{predicted_class}**") 

import os
import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

# ----------------------------
# 🔹 수정된 모델 정의 (checkpoint와 일치)
# ----------------------------
class MyModel(nn.Module):
    def __init__(self, num_classes=4):
        super(MyModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=4, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=4, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(128, 128, kernel_size=4, padding=1)  # 수정됨
        self.bn4 = nn.BatchNorm2d(128)

        self.pool = nn.MaxPool2d(2, 2)
        self.gap = nn.AdaptiveAvgPool2d((6, 6))  # 새로 추가: 6x6 → 4608

        self.fc1 = nn.Linear(128 * 6 * 6, 512)  # 4608 → checkpoint와 일치
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # 224 → 112
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # 112 → 56
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  # 56 → 28
        x = self.pool(F.relu(self.bn4(self.conv4(x))))  # 28 → 14
        x = self.gap(x)                                 # 14x14 → 6x6
        x = x.view(-1, 128 * 6 * 6)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


# ----------------------------
# 🔹 모델 로드 함수
# ----------------------------
def load_model(model_path, device, num_classes=4):
    model = MyModel(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

# ----------------------------
# 🔹 예측 함수
# ----------------------------
def predict(model, image_tensor, device):
    image_tensor = image_tensor.to(device)
    model.eval()
    with torch.no_grad():
        outputs = model(image_tensor)
        probs = F.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probs, dim=1).item()
        probabilities = probs.squeeze().tolist()
    return predicted_class, probabilities

# ----------------------------
# 🔹 이미지 전처리 함수
# ----------------------------
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return transform(image).unsqueeze(0)

# ----------------------------
# 🔹 샘플 이미지 로더
# ----------------------------
@st.cache_data
def load_sample_images(sample_images_dir):
    sample_image_files = os.listdir(sample_images_dir)
    sample_images = []
    for sample_image_file in sample_image_files:
        sample_image_path = os.path.join(sample_images_dir, sample_image_file)
        sample_image = Image.open(sample_image_path).convert("RGB")
        sample_image = sample_image.resize((150, 150))
        sample_images.append((sample_image_file, sample_image))
    return sample_images

# ----------------------------
# 🔹 클래스 라벨 정의 (model_15 기준 4-class)
# ----------------------------
label_dict = {
    0: "Pituitary",
    1: "Glioma",
    2: "Meningioma",
    3: "No Tumor",
}

# ----------------------------
# 🔹 Streamlit 앱 시작
# ----------------------------
st.title("Brain Tumor Classification")

# 모델 로드
device = "cuda" if torch.cuda.is_available() else "cpu"
model_path = os.path.join("models", "model_15")
model = load_model(model_path, device, num_classes=len(label_dict))

# 샘플 이미지 표시
st.subheader("Sample Images")
st.write("Here are some sample images. Your uploaded image should be similar to these for best results.")

sample_images_dir = "sample"
sample_images = load_sample_images(sample_images_dir)
cols = st.columns(3)
for i, (sample_image_file, sample_image) in enumerate(sample_images):
    col_idx = i % 3
    with cols[col_idx]:
        st.image(sample_image, caption=f"Sample {i+1}", use_container_width=True)

# 이미지 업로드
st.write("Upload an image below to classify it.")
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", width=210)

    # 예측
    preprocessed_image = preprocess_image(image).to(device)
    predicted_class, probabilities = predict(model, preprocessed_image, device)

    # 결과 출력
    st.write(
        f"<h1 style='font-size: 48px;'>Prediction: {label_dict[predicted_class]}</h1>",
        unsafe_allow_html=True,
    )

    st.subheader("Prediction Probabilities")
    for i, prob in enumerate(probabilities):
        st.markdown(
            f"<p><b>{label_dict[i]}</b>: {prob * 100:.2f}%</p>",
            unsafe_allow_html=True
        )

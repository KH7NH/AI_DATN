# 🧠 AI Sensitive Image Classifier (EfficientNet-B3)

Project to train **sensitive image classification (Sensitive Image Detection)** model using TensorFlow/Keras using **EfficientNet-B3** architecture.

## ⚙️ Environment settings

### 1️⃣ Create virtual environment and install dependencies

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

2️⃣ Dataset structure
dataset/
├── train/
│   ├── normal/
│   └── sensitive/
└── val/
    ├── normal/
    └── sensitive/

Technology used

TensorFlow / Keras

NumPy, scikit-learn, Matplotlib, Seaborn

EfficientNet-B3 Pretrained (ImageNet weights)

🧑‍💻 Author

Hoang Duc Khanh
# 🧠 AI Sensitive Image Classifier (EfficientNet-B3)

Project to train **sensitive image classification (Sensitive Image Detection)** model using TensorFlow/Keras using **EfficientNet-B3** architecture.

## ⚙️ Environment settings


Create virtual environment and install dependencies

python -m venv venv

venv\Scripts\activate

pip install -r requirements.txt

Dataset structure

dataset/

├── train/

   ├── normal/

   └── sensitive/

└── val/

   ├── normal/

   └── sensitive/

🧩 Model Information

Architecture: EfficientNet-B3 (pretrained on ImageNet)

Framework: TensorFlow / Keras

Loss Function: Binary Cross-Entropy

Optimizer: Adam (learning_rate=5e-6)

Augmentation: rotation, zoom, brightness, flips, etc.

Input size: 300×300×3

Output: Binary classification (Normal / Sensitive)

EfficientNet-B3 Pretrained (ImageNet weights)

📈 Training Details

Epochs: 40

Accuracy achieved: ~87% (Validation)

Loss curve: Stable convergence after epoch 20

Balanced dataset: Class weights applied using compute_class_weight

💾 Pretrained Model

You can download the pretrained model directly from Google Drive:

🔗 Download EfficientNet-B3 Model (Google Drive) 

https://drive.google.com/drive/folders/1KTeXZI9zlBfPBRndxSWbvbtWYImhNWFd?usp=sharing

Model files:

efficientnet_b3_best.keras – Best validation accuracy

efficientnet_b3_final.keras – Final trained version

🧑‍💻 Author

Hoang Duc Khanh
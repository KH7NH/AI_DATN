import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

# ================================
# LOAD MODEL 1 LẦN DUY NHẤT
# ================================

print("🔺 Loading GORE model...")
GORE_MODEL = load_model("models/efficientnet_b3_final.keras", compile=False)
print("✅ GORE model ready!")

print("🔺 Loading NSFW model...")
NSFW_MODEL = load_model("models/nsfw_mobilenetv2_3class.h5", compile=False)
print("✅ NSFW model ready!")


# ================================
# HÀM XỬ LÝ ẢNH
# ================================

def preprocess(img_path, size):
    img = image.load_img(img_path, target_size=size)
    arr = image.img_to_array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr


# ================================
# HÀM PREDICT 2 MODEL
# ================================

def predict_both(img_path):
    gore_input = preprocess(img_path, (300, 300))
    nsfw_input = preprocess(img_path, (224, 224))

    gore_score = float(GORE_MODEL.predict(gore_input)[0][0])
    nsfw_probs = NSFW_MODEL.predict(nsfw_input)[0].tolist()

    return gore_score, nsfw_probs

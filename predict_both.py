import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

# 🔧 Đường dẫn model
GORE_MODEL_PATH = r"models\efficientnet_b3_final.keras"
NSFW_MODEL_PATH = r"models\nsfw_mobilenetv2_3class.h5"

# 🔧 Kích thước ảnh đầu vào cho mỗi model
GORE_IMG_SIZE = (300, 300)   # như script predict_single_image.py của bạn
NSFW_IMG_SIZE = (224, 224)   # sửa lại nếu lúc train NSFW bạn dùng size khác

# 🔧 Ngưỡng để kết luận
GORE_THRESHOLD = 0.5          # > 0.5 coi là GORE
NSFW_LABELS = ["neutral", "nsfw", "sexy"]  # mapping index → nhãn


def load_and_preprocess(img_path, target_size):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def main():
    if len(sys.argv) < 2:
        print("⚠ Usage: python predict_both.py <image_path>")
        sys.exit(1)

    img_path = sys.argv[1]

    print("=== BẮT ĐẦU PREDICT 2 MODEL ===")
    print(f"📸 Ảnh cần predict: {img_path}")

    # ---------- GORE ----------
    print("\n🔺 Loading GORE model ...")
    gore_model = load_model(GORE_MODEL_PATH, compile=False)
    print("✅ GORE model loaded.")

    gore_input = load_and_preprocess(img_path, GORE_IMG_SIZE)
    gore_pred = gore_model.predict(gore_input)
    gore_score = float(gore_pred[0][0])   # 1 neuron sigmoid

    is_gore = gore_score >= GORE_THRESHOLD
    print(f"🔥 GORE score = {gore_score:.4f}  →  {'GORE' if is_gore else 'NORMAL'}")

    # ---------- NSFW ----------
    print("\n🔺 Loading NSFW model ...")
    nsfw_model = load_model(NSFW_MODEL_PATH, compile=False)
    print("✅ NSFW model loaded.")

    nsfw_input = load_and_preprocess(img_path, NSFW_IMG_SIZE)
    nsfw_pred = nsfw_model.predict(nsfw_input)

    nsfw_probs = nsfw_pred[0]               # vector 3 phần tử
    nsfw_idx = int(np.argmax(nsfw_probs))   # class có xác suất cao nhất
    nsfw_label = NSFW_LABELS[nsfw_idx]

    print(f"🔞 NSFW probs = {nsfw_probs.tolist()}")
    print(f"🔞 NSFW label = {nsfw_label}")

    # ---------- TỔNG KẾT ----------
    print("\n===== 🎯 KẾT QUẢ CUỐI CÙNG =====")
    print(f"🔥 GORE: score = {gore_score:.4f}  →  {'GORE' if is_gore else 'NORMAL'}")
    print(f"🔞 NSFW: {nsfw_probs.tolist()}  →  {nsfw_label}")
    print("================================")


if __name__ == "__main__":
    main()

import sys
from pathlib import Path

import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

# ================================
# CẤU HÌNH
# ================================

# Đường dẫn model (tương đối từ thư mục AI)
GORE_MODEL_PATH = Path("models/efficientnet_b3_final.keras")
NSFW_MODEL_PATH = Path("models/nsfw_mobilenetv2_3class.h5")

# Kích thước input
GORE_IMG_SIZE = (300, 300)   # model gore
NSFW_IMG_SIZE = (224, 224)   # model nsfw (sửa nếu bạn train size khác)

# Ngưỡng & label
GORE_THRESHOLD = 0.5
NSFW_LABELS = ["neutral", "nsfw", "sexy"]
NSFW_FLAG_THRESHOLD = 0.7    # min prob để coi nsfw/sexy là đáng báo động

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ================================
# HÀM TIỆN ÍCH
# ================================

def preprocess(img_path: Path, size):
    img = image.load_img(str(img_path), target_size=size)
    arr = image.img_to_array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr


def load_models():
    print("🔺 Đang load GORE model...")
    gore_model = load_model(str(GORE_MODEL_PATH), compile=False)
    print("✅ GORE model sẵn sàng!")

    print("🔺 Đang load NSFW model...")
    nsfw_model = load_model(str(NSFW_MODEL_PATH), compile=False)
    print("✅ NSFW model sẵn sàng!")

    return gore_model, nsfw_model


def predict_one(img_path: Path, gore_model, nsfw_model):
    """Predict 1 ảnh với quy tắc:
       Nếu gore>0.5 hoặc nsfw>0.5 hoặc sexy>0.5 => Nhạy cảm
    """

    # ----- GORE -----
    gore_input = preprocess(img_path, GORE_IMG_SIZE)
    gore_score = float(gore_model.predict(gore_input)[0][0])

    # ----- NSFW -----
    nsfw_input = preprocess(img_path, NSFW_IMG_SIZE)
    nsfw_probs = nsfw_model.predict(nsfw_input)[0]

    # Lấy từng nhãn
    neutral = float(nsfw_probs[0])
    nsfw = float(nsfw_probs[1])
    sexy = float(nsfw_probs[2])

    # Quy tắc kết luận NHẠY CẢM
    is_sensitive = (
        gore_score > 0.5
        or nsfw > 0.5
        or sexy > 0.5
    )

    # Kết luận
    if is_sensitive:
        conclusion = "⚠ NHẠY CẢM (gore/nsfw/sexy vượt ngưỡng)"
    else:
        conclusion = "✅ AN TOÀN"

    # Nhãn cao nhất
    idx = int(np.argmax(nsfw_probs))
    nsfw_label = NSFW_LABELS[idx]
    nsfw_conf = float(nsfw_probs[idx])

    return gore_score, nsfw_probs, nsfw_label, nsfw_conf, conclusion



def collect_images_from_input(path_str: str):
    """
    Nhập 1 path từ người dùng:
    - Nếu là file ảnh → trả về [file]
    - Nếu là folder   → trả về list ảnh trong folder
    """
    base = Path.cwd()
    p = Path(path_str)
    if not p.is_absolute():
        p = base / p

    if p.is_dir():
        files = [
            f for f in sorted(p.iterdir())
            if f.is_file() and f.suffix.lower() in IMG_EXTS
        ]
        return files
    elif p.is_file() and p.suffix.lower() in IMG_EXTS:
        return [p]
    else:
        print(f"⚠ '{p}' không phải file/folder ảnh hợp lệ.")
        return []


# ================================
# MAIN MENU
# ================================

def main():
    print("=== AI CONSOLE – GORE + NSFW ===")

    # 1) Load 2 model MỘT LẦN DUY NHẤT
    gore_model, nsfw_model = load_models()

    while True:
        print("\n===== MENU =====")
        print("1. Predict 1 ảnh")
        print("2. Predict nhiều ảnh (file hoặc folder)")
        print("q. Thoát")
        choice = input("👉 Chọn: ").strip().lower()

        if choice == "q":
            print("👋 Thoát chương trình.")
            break

        elif choice == "1":
            img_path_str = input("Nhập đường dẫn ảnh: ").strip()
            img_path = Path(img_path_str)
            if not img_path.is_absolute():
                img_path = Path.cwd() / img_path

            if not img_path.exists():
                print("⚠ Ảnh không tồn tại.")
                continue

            gore_score, nsfw_probs, nsfw_label, nsfw_conf, conclusion = predict_one(
                img_path, gore_model, nsfw_model
            )

            print("\n===== KẾT QUẢ =====")
            print(f"🖼 Ảnh: {img_path}")
            print(f"🔥 GORE score : {gore_score:.4f} "
                  f"→ {'GORE' if gore_score >= GORE_THRESHOLD else 'NORMAL'}")
            print(f"🔞 NSFW probs : {nsfw_probs.tolist()} "
                  f"→ {nsfw_label} ({nsfw_conf:.4f})")
            print(f"🧠 Kết luận   : {conclusion}")
            print("===================\n")

        elif choice == "2":
            path_str = input("Nhập đường dẫn file hoặc folder chứa ảnh: ").strip()
            img_files = collect_images_from_input(path_str)
            if not img_files:
                continue

            print(f"📂 Số ảnh sẽ predict: {len(img_files)}")
            for f in img_files:
                gore_score, nsfw_probs, nsfw_label, nsfw_conf, conclusion = predict_one(
                    f, gore_model, nsfw_model
                )

                print("\n🖼 Ảnh:", f)
                print(f"   🔥 GORE score : {gore_score:.4f} "
                      f"→ {'GORE' if gore_score >= GORE_THRESHOLD else 'NORMAL'}")
                print(f"   🔞 NSFW probs : {nsfw_probs.tolist()} "
                      f"→ {nsfw_label} ({nsfw_conf:.4f})")
                print(f"   🧠 Kết luận   : {conclusion}")
                print("   ---------------------------------------")

            print("\n✅ DONE batch.\n")

        else:
            print("⚠ Lựa chọn không hợp lệ, hãy nhập 1 / 2 / q.")


if __name__ == "__main__":
    main()

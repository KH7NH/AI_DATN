import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix

# -----------------------------------------------------
# 1️⃣ Cấu hình đường dẫn
# -----------------------------------------------------
BASE_DIR = r"D:\AI\dataset"
MODEL_PATH = r"D:\AI\models\efficientnet_b3_best.keras"   # 🔥 Đổi nếu bạn test model khác
VAL_DIR = os.path.join(BASE_DIR, "val")
OUTPUT_DIR = os.path.join(os.path.dirname(MODEL_PATH), "evaluation_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------------------------------
# 2️⃣ Tải mô hình đã huấn luyện
# -----------------------------------------------------
print(f"🔄 Đang tải mô hình từ: {MODEL_PATH}")
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Mô hình đã được tải thành công!")

# -----------------------------------------------------
# 3️⃣ Chuẩn bị dữ liệu validation/test
# -----------------------------------------------------
IMG_SIZE = (300, 300)
BATCH_SIZE = 32

datagen = ImageDataGenerator(rescale=1.0 / 255)

val_gen = datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

class_indices = val_gen.class_indices
classes = list(class_indices.keys())
print(f"\n📊 Class mapping: {class_indices}")

# -----------------------------------------------------
# 4️⃣ Dự đoán và đánh giá
# -----------------------------------------------------
print("\n🔮 Đang dự đoán trên tập validation...")
pred_probs = model.predict(val_gen, verbose=1)
pred_classes = (pred_probs > 0.5).astype("int32").flatten()
true_classes = val_gen.classes
filenames = val_gen.filenames

# -----------------------------------------------------
# 5️⃣ Báo cáo kết quả
# -----------------------------------------------------
report = classification_report(true_classes, pred_classes, target_names=classes, digits=4)
print("\n📋 Báo cáo đánh giá:")
print(report)

# Lưu ra file
report_path = os.path.join(OUTPUT_DIR, "evaluation_report.txt")
with open(report_path, "w", encoding="utf-8") as f:
    f.write(report)
print(f"📝 Báo cáo đã lưu tại: {report_path}")

# -----------------------------------------------------
# 6️⃣ Confusion Matrix trực quan
# -----------------------------------------------------
cm = confusion_matrix(true_classes, pred_classes)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.tight_layout()

cm_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
plt.savefig(cm_path)
plt.show()
print(f"📊 Confusion matrix đã lưu tại: {cm_path}")

# -----------------------------------------------------
# 7️⃣ Accuracy tổng thể
# -----------------------------------------------------
acc = np.sum(true_classes == pred_classes) / len(true_classes)
print(f"\n✅ Độ chính xác tổng thể: {acc * 100:.2f}%")

# -----------------------------------------------------
# 8️⃣ Hiển thị và lưu ảnh dự đoán sai
# -----------------------------------------------------
print("\n🔍 Đang trích xuất các ảnh dự đoán sai...")

# Tìm chỉ số ảnh dự đoán sai
wrong_indices = np.where(pred_classes != true_classes)[0]

# Tạo thư mục lưu
wrong_dir = os.path.join(OUTPUT_DIR, "misclassified")
os.makedirs(wrong_dir, exist_ok=True)

# Lưu tối đa 20 ảnh minh họa
max_display = 20
if len(wrong_indices) == 0:
    print("🎉 Không có ảnh nào bị dự đoán sai!")
else:
    print(f"⚠️ Có {len(wrong_indices)} ảnh bị dự đoán sai. Hiển thị và lưu tối đa {max_display} ảnh đầu tiên...")

    plt.figure(figsize=(15, 10))
    for i, idx in enumerate(wrong_indices[:max_display]):
        img_path = os.path.join(VAL_DIR, filenames[idx])
        img = tf.keras.preprocessing.image.load_img(img_path, target_size=IMG_SIZE)
        img_arr = tf.keras.preprocessing.image.img_to_array(img) / 255.0

        plt.subplot(4, 5, i + 1)
        plt.imshow(img_arr)
        plt.axis("off")

        true_label = classes[int(true_classes[idx])]
        pred_label = classes[int(pred_classes[idx])]
        conf = pred_probs[idx][0]

        title = f"T:{true_label}\nP:{pred_label}\n({conf:.2f})"
        color = "red" if true_label != pred_label else "green"
        plt.title(title, color=color, fontsize=9)

        # Sao chép ảnh vào thư mục "misclassified"
        save_path = os.path.join(wrong_dir, f"{i+1:02d}_{os.path.basename(img_path)}")
        tf.keras.preprocessing.image.save_img(save_path, img_arr)

    plt.tight_layout()
    wrong_img_path = os.path.join(OUTPUT_DIR, "misclassified_preview.png")
    plt.savefig(wrong_img_path)
    plt.show()
    print(f"🖼️ Ảnh dự đoán sai đã lưu tại: {wrong_img_path}")
    print(f"📂 Toàn bộ ảnh sai được lưu trong thư mục: {wrong_dir}")

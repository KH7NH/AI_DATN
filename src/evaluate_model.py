import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix

BASE_DIR = r"D:\AI\dataset"
MODEL_PATH = r"D:\AI\models\efficientnet_b3_best.keras"   
VAL_DIR = os.path.join(BASE_DIR, "val")
OUTPUT_DIR = os.path.join(os.path.dirname(MODEL_PATH), "evaluation_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load the trained model
print(f"🔄 Loading model from: {MODEL_PATH}")
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ The model has been loaded successfully!")

# Prepare validation/test data
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

# Prediction and assessment
print("\n🔮 Predicting on validation set...")
pred_probs = model.predict(val_gen, verbose=1)
pred_classes = (pred_probs > 0.5).astype("int32").flatten()
true_classes = val_gen.classes
filenames = val_gen.filenames

# Report results
report = classification_report(true_classes, pred_classes, target_names=classes, digits=4)
print("\n📋 Evaluation report:")
print(report)

# Save to file
report_path = os.path.join(OUTPUT_DIR, "evaluation_report.txt")
with open(report_path, "w", encoding="utf-8") as f:
    f.write(report)
print(f"📝 Report saved at: {report_path}")

# Confusion Matrix 
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
print(f"📊 Confusion matrix saved to: {cm_path}")

# Accuracy 
acc = np.sum(true_classes == pred_classes) / len(true_classes)
print(f"\n✅ Overall Accuracy: {acc * 100:.2f}%")

# Display and save wrong prediction images
print("\n🔍 Extracting wrong prediction images...")

# Find the index of the wrong predicted image
wrong_indices = np.where(pred_classes != true_classes)[0]

# Create save folder
wrong_dir = os.path.join(OUTPUT_DIR, "misclassified")
os.makedirs(wrong_dir, exist_ok=True)

# Save up to 20 illustrations
max_display = 20
if len(wrong_indices) == 0:
    print("🎉 No wrongly predicted photos!")
else:
    print(f"⚠️ There are {len(wrong_indices)} incorrectly predicted images. Display and save at most the first {max_display} images...")

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

        # Copy the image to the "misclassified" folder
        save_path = os.path.join(wrong_dir, f"{i+1:02d}_{os.path.basename(img_path)}")
        tf.keras.preprocessing.image.save_img(save_path, img_arr)

    plt.tight_layout()
    wrong_img_path = os.path.join(OUTPUT_DIR, "misclassified_preview.png")
    plt.savefig(wrong_img_path)
    plt.show()
    print(f"🖼️ The wrong prediction photo was saved at: {wrong_img_path}")
    print(f"📂 All the wrong images are saved in the folder: {wrong_dir}")

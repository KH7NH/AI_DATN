import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------------------------------
# 1️⃣ Đường dẫn dữ liệu
# -----------------------------------------------------
BASE_DIR = r"D:\AI\dataset"
TRAIN_DIR = os.path.join(BASE_DIR, "train")
VAL_DIR = os.path.join(BASE_DIR, "val")
MODEL_DIR = os.path.join(r"D:\AI", "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# -----------------------------------------------------
# 2️⃣ Data Augmentation mạnh mẽ hơn
# -----------------------------------------------------
IMG_SIZE = (300, 300)
BATCH_SIZE = 32

train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=30,
    width_shift_range=0.25,
    height_shift_range=0.25,
    shear_range=0.2,
    zoom_range=0.3, 
    brightness_range=[0.7, 1.3],
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1.0/255)

train_gen = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

val_gen = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

print("\n📊 Lớp dữ liệu:", train_gen.class_indices)

# -----------------------------------------------------
# 3️⃣ Cân bằng lớp
# -----------------------------------------------------
y_train = train_gen.classes
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weights = {i: w for i, w in enumerate(class_weights)}
print("⚖️ Class Weights:", class_weights)

# -----------------------------------------------------
# 4️⃣ Xây dựng EfficientNetB0 (fine-tune sâu hơn)
# -----------------------------------------------------
base_model = EfficientNetB3(weights='imagenet', include_top=False, input_shape=(300, 300, 3))

# Giữ lại 60 layer đầu, fine-tune từ layer 60 trở đi
for layer in base_model.layers[:60]:
    layer.trainable = False
for layer in base_model.layers[60:]:
    layer.trainable = True

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
x = Dropout(0.4)(x)
x = Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
x = Dropout(0.4)(x)
x = Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
x = Dropout(0.3)(x)
output = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=output)

optimizer = Adam(learning_rate=5e-6)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

model.summary()

# -----------------------------------------------------
# 5️⃣ Callbacks
# -----------------------------------------------------
checkpoint_path = os.path.join(MODEL_DIR, "efficientnet_b3_best.keras")

checkpoint = ModelCheckpoint(
    checkpoint_path,
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

earlystop = EarlyStopping(
    monitor='val_accuracy',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-7,
    verbose=1
)

csv_logger = CSVLogger(os.path.join(MODEL_DIR, "training_log.csv"), append=False)

# -----------------------------------------------------
# 6️⃣ Huấn luyện
# -----------------------------------------------------
history = model.fit(
    train_gen,
    epochs=40,
    validation_data=val_gen,
    class_weight=class_weights,
    callbacks=[checkpoint, earlystop, reduce_lr, csv_logger]
)

# -----------------------------------------------------
# 7️⃣ Lưu mô hình cuối cùng
# -----------------------------------------------------
final_model_path = os.path.join(MODEL_DIR, "efficientnet_b3_final.keras")
model.save(final_model_path)
print(f"\n✅ Huấn luyện hoàn tất! Mô hình đã được lưu tại: {final_model_path}")

# -----------------------------------------------------
# 8️⃣ Vẽ biểu đồ và lưu lại
# -----------------------------------------------------
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(epochs, acc, 'b-', label='Training acc')
plt.plot(epochs, val_acc, 'r-', label='Validation acc')
plt.title('Training vs Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)

plt.subplot(1,2,2)
plt.plot(epochs, loss, 'b-', label='Training loss')
plt.plot(epochs, val_loss, 'r-', label='Validation loss')
plt.title('Training vs Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, "training_plot.png"))
plt.show()

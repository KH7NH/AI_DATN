import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

MODEL_PATH = "D:\AI\models\efficientnet_b3_final.keras"

model = load_model(MODEL_PATH)
img_path = r"D:\AI\test\sample.jpg"  

img = image.load_img(img_path, target_size=(300, 300))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

prediction = model.predict(img_array)[0][0]

if prediction > 0.5:
    print(f"⚠️  Ảnh này được dự đoán là SENSITIVE ({prediction:.2f})")
else:
    print(f"✅  Ảnh này được dự đoán là NORMAL ({1 - prediction:.2f})")

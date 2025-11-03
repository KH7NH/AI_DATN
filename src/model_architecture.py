from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam

def build_efficientnet_model(img_size=(224, 224, 3), num_classes=1):
    """
    Create model EfficientNetB0 Fine-tuning
    """
    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=img_size)
    base_model.trainable = False  
    
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dropout(0.4)(x)
    output = Dense(num_classes, activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=output)

    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model

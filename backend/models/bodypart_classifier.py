import tensorflow as tf
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import (
    Dense, GlobalAveragePooling2D, Input, Lambda,
    Conv2D, BatchNormalization, Dropout
)
from tensorflow.keras.models import Model
import cv2
import numpy as np

def create_bodypart_model(num_classes):
    """
    Create an improved model using DenseNet121 with custom medical imaging layers
    """
    # Create input layer for grayscale
    inputs = Input(shape=(224, 224, 1))
    
    # Create custom preprocessing branch
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    
    # Convert to 3 channels for DenseNet
    x = Conv2D(3, (1, 1))(x)
    
    # Load and setup DenseNet
    base_model = DenseNet121(
        include_top=False,
        weights='imagenet',
        input_shape=(224, 224, 3)
    )
    
    # Unfreeze some layers for fine-tuning
    for layer in base_model.layers[-50:]:
        layer.trainable = True
    
    # Connect to base model
    x = base_model(x)
    
    # Add custom classification layers
    x = GlobalAveragePooling2D()(x)
    
    # First dense block with residual
    dense1 = Dense(1024, activation='relu')(x)
    dense1 = BatchNormalization()(dense1)
    dense1 = Dropout(0.5)(dense1)
    
    # Second dense block
    dense2 = Dense(512, activation='relu')(dense1)
    dense2 = BatchNormalization()(dense2)
    dense2 = Dropout(0.4)(dense2)
    
    # Final classification
    x = Dense(256, activation='relu')(dense2)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    predictions = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=predictions)
    
    # Use AdamW optimizer for better regularization
    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=0.0001,
        weight_decay=1e-5
    )
    
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Custom layer for medical image enhancement
class MedicalImageEnhancement(tf.keras.layers.Layer):
    def __init__(self, filters, **kwargs):
        super(MedicalImageEnhancement, self).__init__(**kwargs)
        self.filters = filters
        
    def build(self, input_shape):
        # Enhancement convolutions
        self.conv1 = Conv2D(self.filters, (3, 3), padding='same', activation='relu')
        self.conv2 = Conv2D(self.filters, (3, 3), padding='same', activation='relu')
        self.bn1 = BatchNormalization()
        self.bn2 = BatchNormalization()
        
    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        # Add skip connection
        if inputs.shape[-1] == self.filters:
            x = x + inputs
        return x
        
    def get_config(self):
        config = super(MedicalImageEnhancement, self).get_config()
        config.update({"filters": self.filters})
        return config

def preprocess_image(image_path):
    """
    Preprocess an image for the body part classifier model
    """
    # Read image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    # Resize to model input size
    image = cv2.resize(image, (224, 224))
    
    # Normalize to [0,1]
    image = image.astype(np.float32) / 255.0
    
    # Add channel dimension
    image = np.expand_dims(image, axis=-1)
    
    # Add batch dimension
    image = np.expand_dims(image, axis=0)
    
    return image
    
    # Add attention mechanism
    x = tf.expand_dims(x, axis=1)  # Add sequence dimension for attention
    x = AttentionBlock(num_heads=4, key_dim=128)(x)
    x = tf.squeeze(x, axis=1)
    
    # Add attention mechanism
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    x = AttentionBlock(num_heads=4, key_dim=128)(tf.expand_dims(x, axis=1))
    x = tf.squeeze(x, axis=1)
    
    # Improved classification layers with residual connections
    dense1 = Dense(1024, activation='relu')(x)
    dense1 = BatchNormalization()(dense1)
    dense1 = Dropout(0.4)(dense1)
    
    # First residual block
    dense2 = Dense(512, activation='relu')(dense1)
    dense2 = BatchNormalization()(dense2)
    dense2 = Dropout(0.4)(dense2)
    dense2_residual = Dense(512, activation='relu')(dense2)
    dense2 = Add()([dense2, dense2_residual])
    
    # Second residual block
    dense3 = Dense(256, activation='relu')(dense2)
    dense3 = BatchNormalization()(dense3)
    dense3 = Dropout(0.3)(dense3)
    dense3_residual = Dense(256, activation='relu')(dense3)
    dense3 = Add()([dense3, dense3_residual])
    
    # Final classification layers
    x = Dense(128, activation='relu')(dense3)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    
    # Classification head with label smoothing
    predictions = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=predictions)
    
    # Compile with advanced optimizer settings
    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=1e-4,
        weight_decay=1e-5,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-7
    )
    
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=[
            'accuracy',
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.TopKCategoricalAccuracy(k=3, name='top_3_accuracy')
        ]
    )
    
    return model

def preprocess_image(image):
    """
    Enhanced preprocessing pipeline for medical images
    """
    # Ensure grayscale
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Resize
    image = cv2.resize(image, (224, 224))
    
    # Apply CLAHE for better contrast
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    image = clahe.apply(image)
    
    # Apply additional medical image enhancements
    # Denoise with parameters tuned for medical images
    image = cv2.fastNlMeansDenoising(image, None, h=10, templateWindowSize=7, searchWindowSize=21)
    
    # Enhance edges with custom kernel
    kernel = np.array([[-1,-1,-1],
                      [-1, 9,-1],
                      [-1,-1,-1]])
    image = cv2.filter2D(image, -1, kernel)
    
    # Normalize to [0,1]
    image = image.astype('float32')
    image = (image - np.min(image)) / (np.max(image) - np.min(image))
    
    # Add channel dimension if needed
    if len(image.shape) == 2:
        image = np.expand_dims(image, axis=-1)
    
    # Add batch dimension
    image = np.expand_dims(image, axis=0)
    
    return image

def predict_with_confidence(model, image, confidence_threshold=0.7):
    """
    Make predictions with confidence thresholding and ensemble averaging
    """
    # Make base prediction
    pred = model.predict(image)
    
    # Get probabilities and indices
    top_probs = np.sort(pred[0])[-3:][::-1]
    top_indices = np.argsort(pred[0])[-3:][::-1]
    
    # Apply confidence thresholding
    if top_probs[0] < confidence_threshold:
        # If confidence is low, do additional preprocessing and predict again
        # Adjust contrast
        enhanced_image = image.copy()
        enhanced_image = tf.image.adjust_contrast(enhanced_image, 1.5)
        pred2 = model.predict(enhanced_image)
        
        # Average predictions
        final_pred = (pred + pred2) / 2
        top_probs = np.sort(final_pred[0])[-3:][::-1]
        top_indices = np.argsort(final_pred[0])[-3:][::-1]
    
    return top_indices, top_probs
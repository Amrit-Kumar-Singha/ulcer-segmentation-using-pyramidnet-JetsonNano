import tensorflow as tf
import cv2
import os
import numpy as np
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

# Limit GPU memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Constants
HEIGHT, WIDTH = 256, 256
N_CHANNELS = 3
N_CLASSES = 2

def myblock(conv_image, pool_image):
    upsample = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(pool_image)
    concate = concatenate([conv_image, upsample])
    convo = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(concate)
    return convo

def Pyramidnet_model(n_classes=1, IMG_HEIGHT=256, IMG_WIDTH=256, IMG_CHANNELS=3):
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    resnet50 = ResNet50(include_top=False, weights="imagenet", input_tensor=inputs)
    
    c1 = resnet50.get_layer("conv1_relu").output
    p1 = MaxPooling2D((2, 2))(c1)
    
    c2 = resnet50.get_layer("conv2_block1_1_conv").output
    p2 = MaxPooling2D((2, 2))(c2)
    
    c3 = resnet50.get_layer("conv3_block1_1_conv").output
    p3 = MaxPooling2D((2, 2))(c3)
    
    c4 = resnet50.get_layer("conv4_block1_1_conv").output
    p4 = MaxPooling2D((2, 2))(c4)
    
    a1 = myblock(c1, c2)
    a2 = myblock(c2, c3)
    a3 = myblock(c3, c4)
    
    a4 = myblock(a1, a2)
    a5 = myblock(a2, a3)
    
    upsample = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(a5)
    concate = concatenate([a4, upsample])
    
    upsample1 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(concate)
    outputs = Conv2D(2, (1, 1), activation='sigmoid')(upsample1)
    
    model = Model(inputs=[resnet50.input], outputs=[outputs])
    return model

def load_and_preprocess_image(image_path):
    # Read and preprocess the image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    # Store original image for visualization
    original_img = cv2.resize(img, (WIDTH, HEIGHT))
    
    # Normalize the image for model input
    img = cv2.resize(img, (WIDTH, HEIGHT))
    img = img.astype('float32') / 255.0
    
    return np.expand_dims(img, axis=0), original_img

def predict_mask(model, image):
    with tf.device('/GPU:0'):
        prediction = model(image)
        prediction = tf.argmax(prediction, axis=-1)
        prediction = tf.squeeze(prediction, axis=0)
        return tf.expand_dims(prediction, axis=-1)

def save_visualization(original_image, predicted_mask, output_path='prediction_result.png'):
    plt.figure(figsize=(12, 4))
    
    # Display original image
    plt.subplot(131)
    plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    plt.title('Input Image')
    plt.axis('off')
    
    # Display predicted mask
    plt.subplot(132)
    mask_display = predicted_mask.numpy()
    plt.imshow(mask_display, cmap='gray')
    plt.title('Predicted Mask')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Results saved to {output_path}")

def main():
    try:
        # Create and load the model
        print("Creating model...")
        model = Pyramidnet_model(n_classes=N_CLASSES, IMG_HEIGHT=HEIGHT, IMG_WIDTH=WIDTH, IMG_CHANNELS=N_CHANNELS)
        weights_path = 'data/Foot Ulcer Segmentation Challenge/Pyramid_Net-epoch-47.weights.h5'
        
        if not os.path.exists(weights_path):
            raise ValueError(f"Weights file not found at {weights_path}")
            
        print("Loading weights...")
        model.load_weights(weights_path)
        print("Model loaded successfully")

        # Test the model on a single image
        test_image_path = 'data/Foot Ulcer Segmentation Challenge/test/images/1037.png'
        print(f"Processing image: {test_image_path}")
        
        normalized_image, original_image = load_and_preprocess_image(test_image_path)
        print("Making prediction...")
        predicted_mask = predict_mask(model, normalized_image)
        
        # Save the results
        print("Saving visualization...")
        save_visualization(original_image, predicted_mask)
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
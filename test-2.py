import tensorflow as tf
import cv2
import numpy as np
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

# Constants from training
HEIGHT, WIDTH = 256, 256
N_CHANNELS = 3
N_CLASSES = 2
SEED = 42

def parse_image(img_path):
    """
    Load an image and its annotation (mask) and return a dictionary.
    """
    # Reading the image
    image = tf.io.read_file(img_path)
    image = tf.image.decode_jpeg(image, channels=3)

    mask_path = tf.strings.regex_replace(img_path, "images", "labels")
    mask_path = tf.strings.regex_replace(mask_path, ".jpg", ".png")

    # Reading the annotation file
    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=1)
    mask = tf.where(mask == 255, np.dtype('uint8').type(1), mask)

    return {'image': image, 'segmentation_mask': mask}

def normalize(input_image, input_mask):
    """
    Rescale the pixel values of the images between 0 and 1
    """
    input_image = tf.cast(input_image, tf.float32) / 255.0
    return input_image, input_mask

def load_image_test(datapoint):
    input_image = tf.image.resize(datapoint['image'], (HEIGHT, WIDTH))
    input_mask = tf.image.resize(datapoint['segmentation_mask'], (HEIGHT, WIDTH))
    input_image, input_mask = normalize(input_image, input_mask)
    return input_image, input_mask

def display_sample(display_list):
    plt.figure(figsize=(15, 15))
    title = ['Input Image', 'True Mask', 'Predicted Mask']
    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        plt.axis('off')
    plt.savefig('prediction_result.png')
    plt.close()
    print("Results saved as prediction_result.png")

def myblock(conv_image, pool_image):
    upsample = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(pool_image)
    concate = concatenate([conv_image, upsample])
    convo = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(concate)
    return convo

def Pyramidnet_model(n_classes=1, IMG_HEIGHT=256, IMG_WIDTH=256, IMG_CHANNELS=3):
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    resnet50 = ResNet50(include_top=False, weights="imagenet", input_tensor=inputs)
    
    #256
    c1 = resnet50.get_layer("conv1_relu").output
    p1 = MaxPooling2D((2, 2))(c1)
    
    #128
    c2 = resnet50.get_layer("conv2_block1_1_conv").output
    p2 = MaxPooling2D((2, 2))(c2)
    
    #64
    c3 = resnet50.get_layer("conv3_block1_1_conv").output
    p3 = MaxPooling2D((2, 2))(c3)
    
    #32
    c4 = resnet50.get_layer("conv4_block1_1_conv").output
    p4 = MaxPooling2D((2, 2))(c4)
    
    #layer 1
    a1 = myblock(c1, c2)  #c1 and c2
    a2 = myblock(c2, c3)  #c2 and c3
    a3 = myblock(c3, c4)  #c3 and c4
    
    #layer2
    a4 = myblock(a1, a2)  #a1 and a2
    a5 = myblock(a2, a3)  #a2 and a3
    
    upsample = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(a5)
    concate = concatenate([a4, upsample])
    
    upsample1 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(concate)
    outputs = Conv2D(2, (1, 1), activation='sigmoid')(upsample1)
    
    model = Model(inputs=[resnet50.input], outputs=[outputs])
    return model

def predict(model, image_path):
    """
    Predict mask for a single image
    """
    datapoint = parse_image(image_path)
    input_image, image_mask = load_image_test(datapoint)
    img = tf.expand_dims(input_image, 0)
    prediction = model(img)
    prediction = tf.argmax(prediction, axis=-1)
    prediction = tf.squeeze(prediction, axis=0)
    pred_mask = tf.expand_dims(prediction, axis=-1)
    display_sample([input_image, image_mask, pred_mask])
    return image_mask, pred_mask

def calculate_metrics(true_mask, pred_mask):
    """
    Calculate IoU and Dice Score
    """
    # IoU calculation
    IoUs = []
    n_classes = 2
    for c in range(n_classes):
        TP = np.sum((true_mask == c) & (pred_mask == c))
        FP = np.sum((true_mask != c) & (pred_mask == c))
        FN = np.sum((true_mask == c) & (pred_mask != c))
        IoU = TP/float(TP + FP + FN) if (TP + FP + FN) > 0 else 0
        print(f"Class {c} IoU: {IoU:.3f} (TP={TP}, FP={FP}, FN={FN})")
        IoUs.append(IoU)
    mean_iou = np.mean(IoUs)
    print(f"Mean IoU: {mean_iou:.3f}")
    
    # Dice Score calculation
    dice_scores = []
    for c in range(n_classes):
        TP = np.sum((true_mask == c) & (pred_mask == c))
        FP = np.sum((true_mask != c) & (pred_mask == c))
        FN = np.sum((true_mask == c) & (pred_mask != c))
        dice = (2 * TP) / float((2 * TP) + FP + FN) if ((2 * TP) + FP + FN) > 0 else 0
        print(f"Class {c} Dice Score: {dice:.3f}")
        dice_scores.append(dice)
    mean_dice = np.mean(dice_scores)
    print(f"Mean Dice Score: {mean_dice:.3f}")
    
    return mean_iou, mean_dice

def main():
    # Limit GPU memory growth
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
    
    try:
        # Create model
        print("Creating model...")
        model = Pyramidnet_model(n_classes=N_CLASSES, IMG_HEIGHT=HEIGHT, IMG_WIDTH=WIDTH, IMG_CHANNELS=N_CHANNELS)
        
        # Load weights
        weights_path = 'data/Foot Ulcer Segmentation Challenge/Pyramid_Net-epoch-49.weights.h5'
        if not os.path.exists(weights_path):
            raise ValueError(f"Weights file not found at {weights_path}")
        
        print("Loading weights...")
        model.load_weights(weights_path)
        print("Model loaded successfully")
        
        # Test images
        test_images = [
            #'data/Foot Ulcer Segmentation Challenge/train/images/0011.png',
            'data/Foot Ulcer Segmentation Challenge/train/images/0242.png'
        ]
        
        for img_path in test_images:
            print(f"\nProcessing image: {img_path}")
            true_mask, pred_mask = predict(model, img_path)
            mean_iou, mean_dice = calculate_metrics(true_mask.numpy(), pred_mask.numpy())
            print("-" * 50)
            
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
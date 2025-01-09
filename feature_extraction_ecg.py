import cv2
import numpy as np

# Feature extraction function for ECG images
def extract_features_ecg(image_file):
    try:
        img = cv2.imread(image_file)  # Read the image file
        if img is None:
            print(f"Error loading image {image_file}")
            return None  # Return None if the image couldn't be loaded
        
        img_resized = cv2.resize(img, (128, 128))  # Resize to standard size (128x128)
        img_resized = img_resized.astype('float32')  # Normalize pixel values to range [0, 1]
        img_resized /= 255.0
        img_flattened = img_resized.flatten()  # Flatten the image into a 1D array of pixels
        return img_flattened
    except Exception as e:
        print(f"Error extracting features from {image_file}: {e}")
        return None  # Return None in case of an error

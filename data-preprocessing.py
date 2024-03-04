import os
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_and_preprocess_images(directory, target_size = (224, 224)):
    images = []
    labels = []

    # for subdirectoy of labeled image data of finger count
    for label in os.listdir(directory):
        label_dir = os.path.join(directory, label)

        # skip any files in directory and only process in subdirectory folders
        if not os.path.isdir(label_dir):
            continue

    for image_file in os.listdir(label_dir):
        image_path = os.path.join(label_dir, image_file)
        image = cv2.imread(image_path)
import os
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_and_preprocess_images(directory, target_size = (224, 224)):
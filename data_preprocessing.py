import cv2
import os

def load_and_preprocess_fingers(directory, target_size = (224, 224)):
    images = []
    labels = []

    # for subdirectoy of labeled image data of finger count
    for label in os.listdir(directory):
        label_dir = os.path.join(directory, label)

        # skip any files in directory and only process in subdirectory folders
        if not os.path.isdir(label_dir):
            continue

    # goes over every image in subdirectory
    for image_file in os.listdir(label_dir):
        image_path = os.path.join(label_dir, image_file)
        image = cv2.imread(image_path)  #load image into cv2
    
        # check if image is readable then proceed with processing readable images
        if  image is not None:
            image = cv2.resize(image, target_size)
            image = image / 255.0   # stablize pixel value
            image.append(image)
            labels.append(int(label))
    
    return images, labels   #return lists of 'images' and 'data' 


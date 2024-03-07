from data_preprocessing import load_and_preprocess_fingers
from hand_model import create_model
from sklearn.model_selection import train_test_split
import numpy as np

# load and process images
images, labels = load_and_preprocess_fingers('path to dataset')

# convert to arrays
images = np.array(images)
labels = np.array(labels)

# split data for training and validation
X_trian, X_val, y_train, y_val = train_test_split(images, labels, test_size = 0.2, random_state = 42)

# build the mdoel and train it
model = create_model()


model.fit(X_trian, y_train, epochs = 10. validation_data=(X_val, y_val))

# save the model
model.save('finger_mode.h5')
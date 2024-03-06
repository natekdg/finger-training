from tensorflow import Sequential
from tensorflow import Conv2D, MaxPooling2D, Flatten, Dense


# import sequential model from keras
model = Sequential([
    
    #import needed layers
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(6, activation='softmax')
])

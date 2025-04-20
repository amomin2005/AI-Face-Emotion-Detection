from keras.utils import to_categorical
from keras.preprocessing.image import load_img
from keras.models import Sequential 
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from keras.models import model_from_json
from keras.callbacks import ModelCheckpoint  # Import the ModelCheckpoint callback

# Directories
traindirectory = "images/train"
testdirectory = "images/test"

# Function to create dataframe
def createdataframe(dir):
    image_paths = []
    labels = []
    for label in os.listdir(dir):
        for imagename in os.listdir(os.path.join(dir, label)):
            image_paths.append(os.path.join(dir, label, imagename))
            labels.append(label)
        print(label, "completed")
    return image_paths, labels

# Create dataframes for training and testing data
train = pd.DataFrame()
train['image'], train['label'] = createdataframe(traindirectory)

test = pd.DataFrame()
test['image'], test['label'] = createdataframe(testdirectory)

# Feature extraction function
def extractfeatures(images):
    features = []
    for image in tqdm(images):
        img = load_img(image, color_mode='grayscale')
        img = np.array(img)
        features.append(img)
    features = np.array(features)
    features = features.reshape(len(features), 48, 48, 1)
    return features

# Extract features from training and testing data
trainfeatures = extractfeatures(train['image'])
testfeatures = extractfeatures(test['image'])
xtrain = trainfeatures / 255.0
xtest = testfeatures / 255.0

# Label encoding
le = LabelEncoder()
le.fit(train['label'])

ytrain = le.transform(train['label'])
ytest = le.transform(test['label'])
ytrain = to_categorical(ytrain, num_classes=7)
ytest = to_categorical(ytest, num_classes=7)

# Build the model
model = Sequential()
model.add(Conv2D(128, kernel_size=(3,3), activation='relu', input_shape=(48,48,1)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.4))

model.add(Conv2D(256, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.4))

model.add(Conv2D(512, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.4))

model.add(Conv2D(512, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.4))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(7, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Define the checkpoint callback to save the best model during training
checkpoint = ModelCheckpoint('emotiondetector_best_weights.weights.h5', 
                              save_best_only=True, 
                              save_weights_only=True, 
                              verbose=1)

# Try-Except block to handle KeyboardInterrupt and save model
try:
    # Train the model and include the checkpoint callback
    model.fit(x=xtrain, y=ytrain, batch_size=128, epochs=10, validation_data=(xtest, ytest), callbacks=[checkpoint])
except KeyboardInterrupt:
    # Save the model when interrupted
    print("Training interrupted. Saving model...")
    model.save_weights('emotiondetector_best_weights.weights.h5')  # Save the weights
    model_json = model.to_json()  # Save the model architecture
    with open('emotiondetector.json', 'w') as json_file:
        json_file.write(model_json)
    print("Model saved successfully.")

# After training, save the model architecture and final weights
model_json = model.to_json()
with open("emotiondetector.json", 'w') as json_file:
    json_file.write(model_json)

# Save the final weights (in case you want to keep a backup)
model.save_weights("emotiondetector_final_weights.h5")

# Load the model
json_file = open("emotiondetector.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights("emotiondetector_best_weights.weights.h5")

# Prediction function
labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

def ef(image):
    img = load_img(image, color_mode='grayscale')
    feature = np.array(img)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

# Test the model with a sample image
image = 'images/train/sad/42.jpg'
print("Original image is of sad")
img = ef(image)
pred = model.predict(img)
pred_label = labels[pred.argmax()]
print("Model prediction is ", pred_label)
import cv2
from keras.models import model_from_json
import numpy as np

# Load model
json_file = open("emotiondetector.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)

# Load weights
model.load_weights("emotiondetector_best_weights.weights.h5")

# Load Haar cascade for face detection
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

# Function to preprocess the image
def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

# Webcam setup
webcam = cv2.VideoCapture(0)

# Emotion labels
labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

while True:
    # Read frame
    ret, im = webcam.read()
    
    if not ret:
        print("Failed to capture image")
        break

    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(im, 1.3, 5)

    try:
        # Loop through faces
        for (p, q, r, s) in faces:
            # Extract face region
            image = gray[q:q + s, p:p + r]
            
            # Draw rectangle around face
            cv2.rectangle(im, (p, q), (p + r, q + s), (255, 0, 0), 2)
            
            # Resize face image
            image = cv2.resize(image, (48, 48))
            
            # Extract features and predict
            img = extract_features(image)
            pred = model.predict(img)
            
            # Get the label of the prediction
            prediction_label = labels[pred.argmax()]
            
            # Display the label on the image
            cv2.putText(im, '%s' % (prediction_label), (p - 10, q - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255), 2)

        # Show the result
        cv2.imshow("Output", im)

        # Break loop on 'Esc' key press
        if cv2.waitKey(1) & 0xFF == 27:
            break

    except cv2.error as e:
        print(f"Error: {e}")
        pass

# Release the webcam and close windows
webcam.release()
cv2.destroyAllWindows()
"""
Predict the traffic sign class for a single image file.
Useful for testing the model without a webcam (e.g. on a server, in Colab,
or when demoing the project for your CV / a portfolio video).

Usage:
    python predict_image.py path/to/image.png
"""
import sys
import numpy as np
import cv2
from tensorflow.keras.models import load_model

threshold = 0.75  # probability threshold

classNames = [
    "Speed Limit 20 km/h", "Speed Limit 30 km/h", "Speed Limit 50 km/h",
    "Speed Limit 60 km/h", "Speed Limit 70 km/h", "Speed Limit 80 km/h",
    "End of Speed Limit 80 km/h", "Speed Limit 100 km/h", "Speed Limit 120 km/h",
    "No passing", "No passing for vehicles over 3.5 metric tons",
    "Right-of-way at the next intersection", "Priority road", "Yield", "Stop",
    "No vehicles", "Vehicles over 3.5 metric tons prohibited", "No entry",
    "General caution", "Dangerous curve to the left", "Dangerous curve to the right",
    "Double curve", "Bumpy road", "Slippery road", "Road narrows on the right",
    "Road work", "Traffic signals", "Pedestrians", "Children crossing",
    "Bicycles crossing", "Beware of ice/snow", "Wild animals crossing",
    "End of all speed and passing limits", "Turn right ahead", "Turn left ahead",
    "Ahead only", "Go straight or right", "Go straight or left", "Keep right",
    "Keep left", "Roundabout mandatory", "End of no passing",
    "End of no passing by vehicles over 3.5 metric tons",
]


def preprocessing(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img / 255
    return img


def main():
    if len(sys.argv) != 2:
        print("Usage: python predict_image.py path/to/image.png")
        sys.exit(1)

    img_path = sys.argv[1]
    model = load_model("model_trained.keras")

    imgOrignal = cv2.imread(img_path)
    if imgOrignal is None:
        print(f"Could not read image at {img_path}")
        sys.exit(1)

    img = cv2.resize(imgOrignal, (32, 32))
    img = preprocessing(img)
    img = img.reshape(1, 32, 32, 1)

    predictions = model.predict(img, verbose=0)
    classIndex = int(np.argmax(predictions))
    probabilityValue = float(np.amax(predictions))

    print(f"Predicted class: {classIndex} - {classNames[classIndex]}")
    print(f"Confidence: {round(probabilityValue * 100, 2)}%")
    if probabilityValue < threshold:
        print(f"(Below the {int(threshold * 100)}% confidence threshold - prediction may be unreliable)")


if __name__ == "__main__":
    main()
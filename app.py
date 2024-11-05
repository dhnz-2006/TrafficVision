# Import necessary libraries
from flask import Flask, render_template, request, redirect, url_for
import torch
import cv2
import cvzone
from torchvision import transforms
import torchvision
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import requests
import base64
# Initialize the Flask app
app = Flask(__name__)

# Load the pre-trained Faster R-CNN model (update 'pretrained' parameter)
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights

# Update model loading to handle deprecated parameter
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
model.eval()


# Load class names from 'classes.txt'
classnames = []
with open('C:\\Users\\Dhanvanth S\\Documents\\ML\\Faster-RCNN-PYTORCH-main\\classes.txt', 'r') as f:
    classnames = f.read().splitlines()

# Define vehicle categories
vehicle_classes = {
    "car": ["car"],
    "truck": ["truck"],
    "bike": ["bicycle", "motorcycle"],
    "ambulance": ["ambulance"],
    "other_vehicles": ["bus", "train"]
}

# Load historical data for linear regression
data = pd.read_csv('C:\\Users\\Dhanvanth S\\Documents\\ML\\Faster-RCNN-PYTORCH-main\\vehicle_data.csv')
X = data[['car_count', 'truck_count', 'bike_count', 'ambulance_count', 'other_vehicles_count']]
y = data['time_to_pass']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
reg = LinearRegression()
reg.fit(X_train, y_train)

# Define the main route to display the webpage
@app.route('/')
def index():
    return render_template('index.html')

# Define route to process vehicle detection when the button is clicked
@app.route('/process', methods=['POST'])
def process_image():
    # Request image data from ESP32 URL
    esp32_url = "http://192.168.1.7/capture"  # Replace with actual ESP32 URL
    resp = requests.get(esp32_url, stream=True)
    image_data = np.asarray(bytearray(resp.content), dtype="uint8")
    image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
    image = cv2.resize(image, (640, 480))

    # Initialize vehicle count dictionary
    vehicle_count = {
        "car": 0,
        "truck": 0,
        "bike": 0,
        "ambulance": 0,
        "other_vehicles": 0
    }

    # Transform image and run prediction
    image_transform = transforms.ToTensor()
    img = image_transform(image)
    with torch.no_grad():
        pred = model([img])
        bbox, scores, labels = pred[0]['boxes'], pred[0]['scores'], pred[0]['labels']
        conf = torch.argwhere(scores > 0.70).shape[0]

        for i in range(conf):
            x, y, w, h = bbox[i].numpy().astype('int')
            classname_index = labels[i].numpy().astype('int')

            # Check if the classname index is valid
            if classname_index < len(classnames):
                class_detected = classnames[classname_index]
            else:
                print(f"Warning: Detected class index {classname_index} is out of range.")
                continue  # Skip this iteration

            # Categorize and count detected vehicles
            if class_detected in vehicle_classes["car"]:
                vehicle_count["car"] += 1
            elif class_detected in vehicle_classes["truck"]:
                vehicle_count["truck"] += 1
            elif class_detected in vehicle_classes["bike"]:
                vehicle_count["bike"] += 1
            elif class_detected in vehicle_classes["ambulance"]:
                vehicle_count["ambulance"] += 1
            elif class_detected in vehicle_classes["other_vehicles"]:
                vehicle_count["other_vehicles"] += 1

            # Draw bounding boxes on image
            cv2.rectangle(image, (x, y), (w, h), (0, 0, 255), 4)
            cvzone.putTextRect(image, class_detected, [x + 8, y - 12], scale=2, border=1)
    _, buffer = cv2.imencode('.png', image)
    img_base64 = base64.b64encode(buffer).decode('utf-8')

    # Save the annotated image for display
    cv2.imwrite('static/annotated_image.png', image)

    # Predict time to pass based on vehicle count
    vehicle_features = np.array([[vehicle_count['car'], vehicle_count['truck'], vehicle_count['bike'], vehicle_count['ambulance'], vehicle_count['other_vehicles']]])
    
    if sum(sum(vehicle_features)) == 0:
        predicted_time = 0
    else:
        predicted_time = reg.predict(vehicle_features)[0]

    return render_template('result.html', vehicle_count=vehicle_count, predicted_time=predicted_time,img_data=img_base64)


# Run the app
if __name__ == '__main__':
    app.run(debug=True)

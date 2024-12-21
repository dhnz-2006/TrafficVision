from flask import Flask, render_template, request, redirect, url_for
from jinja2 import TemplateNotFound, TemplateSyntaxError
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

app = Flask(__name__)

from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights

try:
    # Load pre-trained model
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    model.eval()
except Exception as e:
    raise RuntimeError("Error loading the model. Ensure you have the necessary libraries installed.") from e

try:
    # Load class names
    classnames = []
    with open('C:\\Users\\Dhanvanth S\\Documents\\ML\\Faster-RCNN-PYTORCH-main\\classes.txt', 'r', encoding='utf-8') as f:
        classnames = f.read().splitlines()
except FileNotFoundError as exc:
    raise FileNotFoundError("classes.txt file not found. Ensure the path is correct.") from exc
except Exception as e:
    raise RuntimeError("An unexpected error occurred while loading class names.") from e

# Define vehicle classes
vehicle_classes = {
    "car": ["car"],
    "truck": ["truck"],
    "bike": ["bicycle", "motorcycle"],
    "ambulance": ["ambulance"],
    "other_vehicles": ["bus", "train"]
}

try:
    # Load CSV data for regression
    data = pd.read_csv('C:\\Users\\Dhanvanth S\\Documents\\ML\\Faster-RCNN-PYTORCH-main\\vehicle_data.csv')
    X = data[['car_count', 'truck_count', 'bike_count', 'ambulance_count', 'other_vehicles_count']]
    y = data['time_to_pass']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    reg = LinearRegression()
    reg.fit(X_train, y_train)
except FileNotFoundError as exc:
    raise FileNotFoundError("vehicle_data.csv file not found. Ensure the path is correct.") from exc
except KeyError as exc:
    raise KeyError("Required columns missing in the CSV file. Check column names.") from exc
except Exception as e:
    raise RuntimeError("An unexpected error occurred while preparing the regression model.") from e

@app.route('/')
def index():
    try:
        return render_template('index.html')
    except FileNotFoundError as exc:
        raise FileNotFoundError("vehicle_data.csv file not found. Ensure the path is correct.") from exc
    except KeyError as exc:
        raise KeyError("Required columns missing in the CSV file. Check column names.") from exc
    except Exception as e:
        raise RuntimeError("An unexpected error occurred while preparing the regression model.") from e


@app.route('/process', methods=['POST'])
def process_image():
    try:
        esp32_url = "http://192.168.1.7/capture"
        resp = requests.get(esp32_url, stream=True, timeout=10)
        resp.raise_for_status()  # Ensure the request was successful
    except requests.exceptions.RequestException as e:
        return f"Error accessing the ESP32 camera API: {str(e)}", 500

    try:
        # Process image data
        image_data = np.asarray(bytearray(resp.content), dtype="uint8")
        image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Failed to decode the image.")
        image = cv2.resize(image, (640, 480))

        vehicle_count = {
            "car": 0,
            "truck": 0,
            "bike": 0,
            "ambulance": 0,
            "other_vehicles": 0
        }

        # Transform and predict
        image_transform = transforms.ToTensor()
        img = image_transform(image)
        with torch.no_grad():
            pred = model([img])
            bbox, scores, labels = pred[0]['boxes'], pred[0]['scores'], pred[0]['labels']
            conf = torch.argwhere(scores > 0.70).shape[0]

            for i in range(conf):
                x, y, w, h = bbox[i].numpy().astype('int')
                classname_index = labels[i].numpy().astype('int')

                if classname_index < len(classnames):
                    class_detected = classnames[classname_index]
                else:
                    continue

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

                cv2.rectangle(image, (x, y), (w, h), (0, 0, 255), 4)
                cvzone.putTextRect(image, class_detected, [x + 8, y - 12], scale=2, border=1)
    except (ValueError, cv2.error, torch.TorchError) as e:
        return f"Error processing the image: {str(e)}", 500

    try:
        # Convert image to base64
        _, buffer = cv2.imencode('.png', image)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        cv2.imwrite('static/annotated_image.png', image)
    except Exception as e:
        return f"Error saving or encoding the annotated image: {str(e)}", 500

    try:
        # Predict time to pass
        vehicle_features = np.array([[vehicle_count['car'], vehicle_count['truck'], vehicle_count['bike'], vehicle_count['ambulance'], vehicle_count['other_vehicles']]])
        if sum(sum(vehicle_features)) == 0:
            predicted_time = 0
        else:
            predicted_time = reg.predict(vehicle_features)[0]
    except (ValueError, KeyError) as e:
        return f"Error predicting time to pass: {str(e)}", 500

    try:
        return render_template('result.html', vehicle_count=vehicle_count, predicted_time=predicted_time, img_data=img_base64)
    except (TemplateNotFound, TemplateSyntaxError) as e:
        return f"Error loading the result page: {str(e)}", 500

if __name__ == '__main__':
    try:
        app.run(debug=True)
    except (OSError, RuntimeError) as e:
        print(f"Error starting the Flask app: {str(e)}")

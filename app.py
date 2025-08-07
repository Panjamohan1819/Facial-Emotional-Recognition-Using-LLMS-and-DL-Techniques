from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from model import EmotionClassifier

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Ensure the upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Define emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EmotionClassifier()
model.load_state_dict(torch.load("saved_models/emotion_classifier.pth", map_location=device))
model.to(device)
model.eval()

# Transform for input image
transform = transforms.Compose([
    transforms.Grayscale(),                # ensure single channel
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return "No file part"

    file = request.files['image']
    if file.filename == '':
        return "No selected file"

    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Open and transform the image
        image = Image.open(file_path).convert('L')  # grayscale
        image = transform(image)
        image = image.unsqueeze(0).to(device)  # add batch dimension

        with torch.no_grad():
            outputs = model(image)
            _, predicted = torch.max(outputs, 1)
            predicted_emotion = emotion_labels[predicted.item()]

        return render_template('result.html', prediction=predicted_emotion, image_path=file_path)

    return "Error during prediction"

if __name__ == '__main__':
    app.run(debug=True)
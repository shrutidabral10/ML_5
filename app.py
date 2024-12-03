import os
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)

# Load your pre-trained model
model = load_model('deepfake_model.keras')


# Preprocessing function for uploaded images
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(128, 128))  # Resize to match model input size
    img_array = image.img_to_array(img)                     # Convert image to array
    img_array = np.expand_dims(img_array, axis=0)           # Add batch dimension
    img_array = img_array / 255.0                           # Normalize pixel values
    return img_array


# Main route for file upload
@app.route('/')
def index():
    return render_template('index.html')


# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return render_template('index.html', error="No file uploaded.")
    file = request.files['image']
    if file.filename == '':
        return render_template('index.html', error="No file selected.")

    # Ensure uploads directory exists
    if not os.path.exists('uploads'):
        os.makedirs('uploads')

    # Save the uploaded file
    file_path = os.path.join('uploads', file.filename)
    file.save(file_path)

    # Preprocess and predict
    processed_img = preprocess_image(file_path)
    prediction = model.predict(processed_img)

    # Remove the file after prediction
    os.remove(file_path)

    # Interpret result
    result = "Deepfake detected" if prediction[0] < 0.5 else "Real image detected"
    color = "red" if "Deepfake" in result else "green"

    return render_template('result.html', prediction=result, color=color)


if __name__ == '__main__':
    app.run(debug=True)


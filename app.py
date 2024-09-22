import os
import json
from PIL import Image
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, redirect, url_for, flash

# Initialize the Flask app
app = Flask(__name__)
app.secret_key = 'some_secret_key'  # You can choose any secret key for Flask session management

# Set the working directory 
model_path = r"D:\CODING\MACHINE LEARNING\MACHINE LEARNING PROJECTS\Plant Health Detection\model.h5"  # Updated model path

class_indices_path = r"D:\CODING\MACHINE LEARNING\MACHINE LEARNING PROJECTS\Plant Health Detection\class_indices.json"  # Update this if necessary

# Load the pre-trained model
model = tf.keras.models.load_model(model_path, compile=False)

# Load the class names
with open(class_indices_path) as f:
    class_indices = json.load(f)

# Function to Load and Preprocess the Image using Pillow
def load_and_preprocess_image(image, target_size=(256, 256)):
    # Resize the image
    img = image.resize(target_size)
    # Convert the image to a numpy array
    img_array = np.array(img)
    # Check if image is grayscale and convert to RGB if necessary
    if img_array.ndim == 2:
        img_array = np.stack([img_array] * 3, axis=-1)
    elif img_array.shape[-1] != 3:
        raise ValueError(f"Expected 3 channels, but got {img_array.shape[-1]}")
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    # Scale the image values to [0, 1]
    img_array = img_array.astype('float32') / 255.0
    return img_array

# Function to Predict the Class of an Image
def predict_image_class(model, image, class_indices):
    preprocessed_img = load_and_preprocess_image(image)
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_indices.get(str(predicted_class_index), 'Unknown class')
    return predicted_class_name

# Define the route for the homepage
@app.route("/", methods=["GET", "POST"])
def upload_image():
    if request.method == "POST":
        # Check if a file was uploaded
        if 'file' not in request.files:
            flash('No file uploaded')
            return redirect(request.url)
        
        file = request.files['file']

        if file.filename == '':
            flash('No file selected for uploading')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            try:
                # Open the uploaded image
                image = Image.open(file).convert('RGB')
                # Predict the class of the image
                prediction = predict_image_class(model, image, class_indices)
                # Render the result page with the prediction
                return render_template('result.html', filename=file.filename, prediction=prediction)
            except Exception as e:
                flash(f"Error processing the image: {e}")
                return redirect(request.url)
    
    return render_template('index.html')

# Allowed file extensions for image uploads
def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Serve the uploaded image
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return redirect(url_for('static', filename='uploads/' + filename), code=301)

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)

from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model('mobilenetv2_house_rooms_model.h5')

# Define the categories
categories = ['Bathroom', 'Bedroom', 'Dinning', 'Kitchen', 'Livingroom', 'Street']

def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes))
    image = image.resize((128, 128))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    image_file = request.files['image']
    image_bytes = image_file.read()
    image = preprocess_image(image_bytes)
    
    predictions = model.predict(image)
    predicted_class = np.argmax(predictions, axis=1)[0]
    predicted_label = categories[predicted_class]
    
    return jsonify({
        'prediction': predicted_label,
        'confidence': float(np.max(predictions))
    })

if __name__ == '__main__':
    app.run(debug=True)

from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import boto3
from keras.models import load_model

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model('mobilenetv2_house_rooms_model_resaved.h5')

# Define the categories
categories = ['BATHROOM', 'BEDROOM', 'DINNING', 'KITCHEN', 'LIVINGROOM', 'STREET']


# S3 Client setup
s3_client = boto3.client(
    's3',
    aws_access_key_id='AKIAQEIP3JJOY4DHINSF',
    aws_secret_access_key='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',
    region_name='ap-southeast-2'
)

def preprocess_image(image_bytes):
    IMAGE_SIZE = (128, 128)
    image = tf.image.decode_jpeg(image_bytes, channels=3)
    image = tf.image.central_crop(image, central_fraction=0.8)
    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.expand_dims(image, axis=0)
    return image

@app.route('/api/image/classify', methods=['POST'])
def classify_image():
    data = request.get_json()
    
    if 'url' not in data:
        return jsonify({'error': 'No URL provided'}), 400

    image_url = data['url']
    bucket_name = 'visioncore-image-bucket' 
    key = image_url.split(f'https://{bucket_name}.s3.amazonaws.com/')[1]


    try:
        image_object = s3_client.get_object(Bucket=bucket_name, Key=key)
        image_bytes = image_object['Body'].read()
    except Exception as e:
        return jsonify({'error': 'Error retrieving image from S3', 'details': str(e)}), 500

    image = preprocess_image(image_bytes)
    predictions = model.predict(image)[0]
    confidence_scores = {category: float(score) for category, score in zip(categories, predictions)}

    return jsonify({
        'confidence_scores': confidence_scores,
        'file_name': image_url
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)

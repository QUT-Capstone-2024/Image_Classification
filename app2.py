#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import mysql.connector
import boto3

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model('mobilenetv2_house_rooms_model.h5')

# Define the categories
categories = ['Bathroom', 'Bedroom', 'Dinning', 'Kitchen', 'Livingroom', 'Street']

# Database connection setup
db = mysql.connector.connect(
    host="localhost",
    user="centralapi_user",
    password="password",
    database="central_api"
)

# S3 Client setup
s3_client = boto3.client(
    's3',
    aws_access_key_id='AKIAQEIP3JJOY4DHINSF',
    aws_secret_access_key='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',
    region_name='ap-southeast-2'
)

def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes))
    image = image.resize((128, 128))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0) 
    return image

@app.route('/api/image/classify', methods=['POST'])
def classify_image():
    if 'image_id' not in request.form:
        return jsonify({'error': 'No image ID provided'}), 400

    image_id = request.form['image_id']
    cursor = db.cursor(dictionary=True)
    cursor.execute("SELECT url FROM image WHERE id = %s", (image_id,))
    image_record = cursor.fetchone()
    cursor.close()

    if not image_record:
        return jsonify({'error': 'Image not found'}), 404
    image_url = image_record['url']
    bucket_name = 'visioncore-image-bucket' 
    key = image_url.split(f'https://{bucket_name}.s3.ap-southeast-2.amazonaws.com/')[1]
    try:
        image_object = s3_client.get_object(Bucket=bucket_name, Key=key)
        image_bytes = image_object['Body'].read()
    except Exception as e:
        return jsonify({'error': 'Error retrieving image from S3', 'details': str(e)}), 500
    image = preprocess_image(image_bytes)
    predictions = model.predict(image)[0]
    confidence_scores = {category: float(score) for category, score in zip(categories, predictions)}

    return jsonify({
        'image_id': image_id,
        'confidence_scores': confidence_scores,
        'file_name' : image_url
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)


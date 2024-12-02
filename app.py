from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
import cv2
import os

app = Flask(__name__)

# Load the pre-trained model
model = tf.keras.models.load_model('model/deepfake_video_model.h5')

IMG_SIZE = 224
MAX_SEQ_LENGTH = 20
NUM_FEATURES = 2048

# Define the feature extractor (InceptionV3)
def build_feature_extractor():
    feature_extractor = tf.keras.applications.InceptionV3(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
    )

    preprocess_input = tf.keras.applications.inception_v3.preprocess_input

    inputs = tf.keras.Input((IMG_SIZE, IMG_SIZE, 3))
    preprocessed = preprocess_input(inputs)
    outputs = feature_extractor(preprocessed)

    return tf.keras.Model(inputs, outputs, name="feature_extractor")

feature_extractor = build_feature_extractor()

# Utility function to load and process video
def load_video(path, max_frames=0, resize=(IMG_SIZE, IMG_SIZE)):
    frames = []
    cap = cv2.VideoCapture(path)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if resize:
            frame = cv2.resize(frame, resize)
        frames.append(frame)
        if max_frames and len(frames) >= max_frames:
            break
    cap.release()
    return np.array(frames)

# Function to crop the center square of a video frame (if needed)
def crop_center_square(frame):
    y, x = frame.shape[0:2]
    min_dim = min(y, x)
    start_x = (x // 2) - (min_dim // 2)
    start_y = (y // 2) - (min_dim // 2)
    return frame[start_y:start_y + min_dim, start_x:start_x + min_dim]

# Utility function to prepare video for prediction
def prepare_single_video(frames):
    frames = frames[None, ...]  # Add batch dimension

    frame_mask = np.zeros(shape=(1, MAX_SEQ_LENGTH), dtype="bool")
    frame_features = np.zeros(shape=(1, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32")

    for i, batch in enumerate(frames):
        video_length = batch.shape[0]  # Corrected: video_length is batch.shape[0]
        length = min(MAX_SEQ_LENGTH, video_length)  # Limit the number of frames

        for j in range(length):
            # Extract features
            frame_features[i, j, :] = feature_extractor.predict(batch[None, j, :])  # Extract features

        frame_mask[i, :length] = 1  # Mark valid frames in the mask

    return frame_features, frame_mask

# Home Route
@app.route('/')
def home():
    return render_template('index.html')

# Endpoint to predict if the video is deepfake or not
@app.route('/predict', methods=['POST'])
def predict():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400

    video = request.files['video']

    # Create a directory to store the uploaded video if it doesn't exist
    if not os.path.exists('uploads'):
        os.makedirs('uploads')

    video_path = os.path.join('uploads', video.filename)

    # Save the video file
    try:
        video.save(video_path)
        print(f"Video saved at {video_path}")
    except Exception as e:
        return jsonify({'error': f'Error saving video: {str(e)}'}), 500

    # Load and process the video
    try:
        print("Loading video...")
        frames = load_video(video_path)
        print(f"Video loaded with {len(frames)} frames")
        frame_features, frame_mask = prepare_single_video(frames)
        print(f"Video features extracted with shape: {frame_features.shape}")
    except Exception as e:
        return jsonify({'error': f'Error processing video: {str(e)}'}), 500

    # Make prediction using the model
    try:
        print("Making prediction...")
        prediction = model.predict([frame_features, frame_mask])
        print(f"Prediction result: {prediction}")
    except Exception as e:
        return jsonify({'error': f'Error making prediction: {str(e)}'}), 500

    # Assuming the model outputs a binary classification (0 for real, 1 for deepfake)
    result = 'Fake' if prediction[0] > 0.5 else 'Real'

    # Convert the prediction result to a list before returning it.
    prediction_result = prediction.tolist()  # Convert numpy.ndarray to list

    # Remove the uploaded video after processing
    try:
        os.remove(video_path)
        print(f"Video removed from {video_path}")
    except Exception as e:
        return jsonify({'error': f'Error removing video file: {str(e)}'}), 500
    
    # Return the result with prediction_values and video_path
    return jsonify({
        'prediction': result,
        'prediction_values': prediction_result,
        'video_path': video_path
    })

# Run the app
if __name__ == '__main__':
    app.run(debug=True)

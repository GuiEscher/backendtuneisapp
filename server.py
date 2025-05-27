#!/usr/bin/env python3

import os
import json
import torch
import cv2 as cv
import numpy as np
from flask import Flask, request, jsonify, Response
from ultralytics import YOLO
from PIL import Image as PILImage
import threading
import time
import io
import tempfile
import zipfile

# Override torch.load to use weights_only=False
orig_torch_load = torch.load
def custom_torch_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return orig_torch_load(*args, **kwargs)
torch.load = custom_torch_load

# Load the trained model
model_path = os.path.join("models", "model.pt")
print(f"Verifying model path: {model_path}")
print(f"Model path exists: {os.path.exists(model_path)}")
print(f"Current working directory: {os.getcwd()}")
print(f"Files in models directory: {os.listdir('models') if os.path.exists('models') else 'models directory not found'}")
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at: {model_path}")

model = YOLO(model_path)

# Class mapping dictionary
label_dict = model.names

# Colors for different classes
class_colors = {
    0: (0, 255, 0),  # Umidade
    1: (0, 0, 255),  # Corrosão
    2: (255, 0, 0),  # Rachadura
}

app = Flask(__name__)

# Global variable for camera state
camera = None
processing_lock = threading.Lock()

def initialize_camera():
    global camera
    camera = cv.VideoCapture(0)
    if not camera.isOpened():
        raise RuntimeError("Failed to open camera")

def process_frame(frame):
    results = model(frame)  # Perform prediction on the frame
    detections = []
    log_message = results[0].verbose() if len(results) > 0 else "No detections"
    
    for result in results:
        for box in result.boxes:
            cls = int(box.cls)  # Detected object class
            conf = float(box.conf)  # Detection confidence
            if conf < 0.3:
                continue  # Ignore low-confidence detections

            # Calculate bounding box coordinates
            x_center, y_center, width, height = map(float, box.xywhn[0])
            width_total = width * frame.shape[1]
            height_total = height * frame.shape[0]

            # Convert normalized coordinates to pixels
            x1 = int((x_center * frame.shape[1]) - width_total / 2)
            y1 = int((y_center * frame.shape[0]) - height_total / 2)
            x2 = int((x_center * frame.shape[1]) + width_total / 2)
            y2 = int((y_center * frame.shape[0]) + height_total / 2)
            
            label = label_dict.get(cls, f"Class {cls}")
            color = class_colors.get(cls, (255, 255, 255))
            label_text = f"{label} ({conf:.2f})"

            # Draw bounding box
            cv.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv.putText(frame, label_text, (x1, y1 - 5), cv.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Process mask with robust checks
            if hasattr(result, 'masks') and result.masks is not None and result.masks.data is not None:
                try:
                    # Get and resize mask
                    mask = result.masks.data[0].cpu().numpy()
                    mask = cv.resize(mask, (frame.shape[1], frame.shape[0]))
                    
                    # Convert to uint8 and binarize
                    mask = (mask * 255).astype(np.uint8)
                    _, binary_mask = cv.threshold(mask, 0.5, 255, cv.THRESH_BINARY)
                    
                    # Create color mask
                    color_mask = np.zeros_like(frame)
                    color_mask[:] = color
                    
                    # Apply mask
                    mask_applied = cv.bitwise_and(color_mask, color_mask, mask=binary_mask)
                    frame = cv.addWeighted(frame, 0.7, mask_applied, 0.3, 0)
                except Exception as e:
                    log_message += f"\nError applying mask: {str(e)}"

            detections.append({
                "class": label,
                "confidence": conf,
                "x_center": x_center,
                "y_center": y_center,
                "width": width,
                "height": height
            })
    
    return frame, detections, log_message

def process_video(video_path):
    cap = cv.VideoCapture(video_path)
    fps = cap.get(cv.CAP_PROP_FPS)
    frame_interval = int(fps * 3)  # Process every 3 seconds
    frame_count = 0
    all_detections = []
    processed_frames = []
    process_logs = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        if frame_count % frame_interval != 0:
            continue
            
        processed_frame, detections, log = process_frame(frame)
        all_detections.extend(detections)
        process_logs.append(log)
        
        # Convert frame to JPEG
        ret, jpeg = cv.imencode('.jpg', processed_frame)
        if ret:
            processed_frames.append(jpeg.tobytes())
    
    cap.release()
    return processed_frames, all_detections, process_logs

@app.route('/capture', methods=['GET'])
def capture():
    global camera
    
    if camera is None:
        initialize_camera()
    
    with processing_lock:
        ret, frame = camera.read()
        if not ret:
            return jsonify({"error": "Failed to capture frame from camera"}), 500
        
        processed_frame, detections, log = process_frame(frame.copy())
        
        # Convert frame to JPEG
        ret, jpeg = cv.imencode('.jpg', processed_frame)
        if not ret:
            return jsonify({"error": "Failed to encode image"}), 500
        
        # Create response with image and JSON data
        img_io = io.BytesIO(jpeg.tobytes())
        
        return Response(
            img_io.getvalue(),
            mimetype='image/jpeg',
            headers={
                'Detections': json.dumps(detections),
                'Logs': json.dumps([log]),
                'Access-Control-Expose-Headers': 'Detections,Logs'
            }
        )

@app.route('/detect', methods=['POST'])
def detect():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    try:
        # Check if it's a video or image
        is_video = file.filename.lower().endswith(('.mp4', '.avi', '.mov'))
        
        if is_video:
            # Save temporary video
            temp_video = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            file.save(temp_video.name)
            temp_video.close()
            
            # Process video
            processed_frames, all_detections, process_logs = process_video(temp_video.name)
            
            # Remove temporary file
            os.unlink(temp_video.name)
            
            if not processed_frames:
                return jsonify({"error": "No frames processed from video"}), 500
                
            # Create a ZIP file with all processed frames
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
                for i, frame in enumerate(processed_frames):
                    zip_file.writestr(f'frame_{i}.jpg', frame)
            
            zip_buffer.seek(0)
            
            return Response(
                zip_buffer.getvalue(),
                mimetype='application/zip',
                headers={
                    'Detections': json.dumps(all_detections),
                    'Logs': json.dumps(process_logs),
                    'Frame-Count': str(len(processed_frames)),
                    'Access-Control-Expose-Headers': 'Detections,Logs,Frame-Count'
                }
            )
        else:
            # Process image
            img_bytes = file.read()
            frame = cv.imdecode(np.frombuffer(img_bytes, np.uint8), cv.IMREAD_COLOR)
            if frame is None:
                return jsonify({"error": "Failed to decode image"}), 400
            
            # Process the frame
            processed_frame, detections, log = process_frame(frame)
            
            # Convert frame to JPEG
            ret, jpeg = cv.imencode('.jpg', processed_frame)
            if not ret:
                return jsonify({"error": "Failed to encode image"}), 500
            
            # Return processed image and detections
            img_io = io.BytesIO(jpeg.tobytes())
            
            return Response(
                img_io.getvalue(),
                mimetype='image/jpeg',
                headers={
                    'Detections': json.dumps(detections),
                    'Logs': json.dumps([log]),
                    'Access-Control-Expose-Headers': 'Detections,Logs'
                }
            )
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"}), 200

def run_server():
    # Create temporary directory
    temp_dir = os.path.join(os.path.dirname(__file__), 'temp')
    os.makedirs(temp_dir, exist_ok=True)
    
    port = int(os.environ.get("PORT", 10000))  # Use Render's PORT or fallback
    print(f"Starting server on port: {port}")
    app.run(host='0.0.0.0', port=port, debug=False)

if __name__ == '__main__':
    run_server()
#!/usr/bin/env python3

from ultralytics import YOLO
import cv2 as cv
import numpy as np
from flask import Flask, request, jsonify, Response
import threading
import time
from werkzeug.serving import make_server
import io
from PIL import Image as PILImage
import os
import tempfile
import zipfile
import json

# Carregando o modelo treinado
model_path = r"D:\tuneisappflutter\tuneisappflutter\server\18-12-24-Model-best (2).pt"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at: {model_path}")

model = YOLO(model_path)

# Dicionário de mapeamento das classes
label_dict = model.names

# Cores para diferenciar cada classe
class_colors = {
    0: (0, 255, 0),    # Umidade
    1: (0, 0, 255),    # Corrosão
    2: (255, 0, 0),    # Rachadura
}

app = Flask(__name__)

# Variável global para armazenar o estado da câmera
camera = None
processing_lock = threading.Lock()

def initialize_camera():
    global camera
    camera = cv.VideoCapture(0)
    if not camera.isOpened():
        raise RuntimeError("Não foi possível abrir a câmera")

def process_frame(frame):
    results = model(frame)  # Realiza a predição no frame
    detections = []
    log_message = results[0].verbose() if len(results) > 0 else "No detections"
    
    for result in results:
        for box in result.boxes:
            cls = int(box.cls)  # Classe do objeto detectado
            conf = float(box.conf)  # Confiança da detecção
            if conf < 0.3:
                continue  # Ignora detecções com baixa confiança

            # Calcular a área da caixa delimitadora
            x_center, y_center, width, height = map(float, box.xywhn[0])
            width_total = width * frame.shape[1]
            height_total = height * frame.shape[0]
            area = width_total * height_total

            # Convertendo coordenadas normalizadas para pixels
            x1 = int((x_center * frame.shape[1]) - width_total / 2)
            y1 = int((y_center * frame.shape[0]) - height_total / 2)
            x2 = int((x_center * frame.shape[1]) + width_total / 2)
            y2 = int((y_center * frame.shape[0]) + height_total / 2)
            
            label = label_dict.get(cls, f"Classe {cls}") 
            color = class_colors.get(cls, (255, 255, 255)) 
            label_text = f"{label} ({conf:.2f})"

            # Desenha a caixa delimitadora
            cv.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv.putText(frame, label_text, (x1, y1 - 5), cv.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Processamento da máscara com verificação mais robusta
            if hasattr(result, 'masks') and result.masks is not None and result.masks.data is not None:
                try:
                    # Obter a máscara e redimensionar para o tamanho do frame
                    mask = result.masks.data[0].cpu().numpy()
                    mask = cv.resize(mask, (frame.shape[1], frame.shape[0]))
                    
                    # Converter para uint8 e binarizar
                    mask = (mask * 255).astype(np.uint8)
                    _, binary_mask = cv.threshold(mask, 0.5, 255, cv.THRESH_BINARY)
                    
                    # Criar máscara de cor
                    color_mask = np.zeros_like(frame)
                    color_mask[:] = color
                    
                    # Aplicar a máscara
                    mask_applied = cv.bitwise_and(color_mask, color_mask, mask=binary_mask)
                    frame = cv.addWeighted(frame, 0.7, mask_applied, 0.3, 0)
                except Exception as e:
                    log_message += f"\nErro ao aplicar máscara: {str(e)}"

            detections.append({
                "class": label,
                "confidence": conf,
                "x_center": x_center,
                "y_center": y_center,
                "width": width,
                "height": height,
                "area": area
            })
    
    return frame, detections, log_message

def process_video(video_path):
    cap = cv.VideoCapture(video_path)
    fps = cap.get(cv.CAP_PROP_FPS)
    frame_interval = int(fps * 3)  # Processar a cada 3 segundos
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
        
        # Converter frame para JPEG
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
            return jsonify({"error": "Falha ao capturar frame da câmera"}), 500
        
        processed_frame, detections, log = process_frame(frame.copy())
        
        # Converter frame para JPEG
        ret, jpeg = cv.imencode('.jpg', processed_frame)
        if not ret:
            return jsonify({"error": "Falha ao codificar imagem"}), 500
        
        # Criar resposta com imagem e dados JSON
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
        return jsonify({"error": "Nenhum arquivo enviado"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "Nenhum arquivo selecionado"}), 400
    
    try:
        # Verifica se é vídeo ou imagem
        is_video = file.filename.lower().endswith(('.mp4', '.avi', '.mov'))
        
        if is_video:
            # Salvar vídeo temporário
            temp_video = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            file.save(temp_video.name)
            temp_video.close()
            
            # Processar vídeo
            processed_frames, all_detections, process_logs = process_video(temp_video.name)
            
            # Remove o arquivo temporário
            os.unlink(temp_video.name)
            
            if not processed_frames:
                return jsonify({"error": "Nenhum frame processado do vídeo"}), 500
                
            # Criar um arquivo ZIP com todos os frames processados
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
            # Processamento de imagem
            img_bytes = file.read()
            frame = cv.imdecode(np.frombuffer(img_bytes, np.uint8), cv.IMREAD_COLOR)
            if frame is None:
                return jsonify({"error": "Falha ao decodificar imagem"}), 400
            
            # Processar o frame
            processed_frame, detections, log = process_frame(frame)
            
            # Converter frame para JPEG
            ret, jpeg = cv.imencode('.jpg', processed_frame)
            if not ret:
                return jsonify({"error": "Falha ao codificar imagem"}), 500
            
            # Retornar imagem processada e detecções
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
    # Cria diretório temporário se não existir
    temp_dir = os.path.join(os.path.dirname(__file__), 'temp')
    os.makedirs(temp_dir, exist_ok=True)
    
    server = make_server('0.0.0.0', 5000, app)
    print("Servidor iniciado em http://0.0.0.0:5000")
    server.serve_forever()

if __name__ == '__main__':
    run_server()
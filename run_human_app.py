# -*- coding: utf-8 -*-
"""
MediaPipe Vision Local Server Launcher with Real-Time Human Detection Prediction API.
Starts a local HTTP server and opens the Web App in the default browser.
Exposes a /predict endpoint to classify cropped face/person images using a trained SVM model.
"""

import os
import sys

# Prevent shadowing the installed mediapipe library
if '' in sys.path:
    sys.path.remove('')
if os.path.dirname(os.path.abspath(__file__)) in sys.path:
    sys.path.remove(os.path.dirname(os.path.abspath(__file__)))

import http.server
import socketserver
import webbrowser
import threading
import time
import json
import base64
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python import BaseOptions

PORT = 8000
DIRECTORY = os.path.dirname(os.path.abspath(__file__))

# Global variables for models
embedder = None
human_svm = None
models_loaded = False
models_error = None

def init_models():
    global embedder, human_svm, models_loaded, models_error
    model_path = os.path.join(DIRECTORY, "models", "mobilenet_v3_small.tflite")
    svm_path = os.path.join(DIRECTORY, "models", "human_svm.xml")
    
    print("\n[*] Servidor: Inicializando modelos em segundo plano...")
    try:
        # Load Image Embedder
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Modelo base não encontrado em {model_path}")
            
        options = vision.ImageEmbedderOptions(
            base_options=BaseOptions(model_asset_path=model_path)
        )
        embedder = vision.ImageEmbedder.create_from_options(options)
        print("[+] Servidor: ImageEmbedder carregado com sucesso.")
        
        # Load SVM
        if os.path.exists(svm_path):
            human_svm = cv2.ml.SVM_load(svm_path)
            print("[+] Servidor: Classificador SVM de Humanos carregado com sucesso.")
            models_loaded = True
        else:
            print("[!] Servidor Aviso: Classificador SVM não encontrado em models/human_svm.xml.")
            print("[!] Por favor, execute train_human_classifier.py para treinar a IA.")
            models_error = "SVM não treinado. Execute o treinamento primeiro."
            
    except Exception as e:
        print(f"[!] Servidor Erro ao carregar modelos: {e}")
        models_error = str(e)

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'X-Requested-With, Content-Type')
        self.end_headers()

    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS, POST')
        self.send_header('Access-Control-Allow-Headers', 'X-Requested-With, Content-Type')
        super().end_headers()

    def do_POST(self):
        if self.path == '/predict':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            
            response_data = {}
            try:
                data = json.loads(post_data.decode('utf-8'))
                img_data = data['image']
                
                if ',' in img_data:
                    img_data = img_data.split(',')[1]
                    
                img_bytes = base64.b64decode(img_data)
                nparr = np.frombuffer(img_bytes, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if img is None:
                    raise ValueError("Falha ao decodificar a imagem enviada.")
                
                global embedder, human_svm, models_loaded, models_error
                
                if human_svm is None:
                    svm_path = os.path.join(DIRECTORY, "models", "human_svm.xml")
                    if os.path.exists(svm_path):
                        human_svm = cv2.ml.SVM_load(svm_path)
                        models_loaded = True
                        models_error = None
                        print("[+] Servidor: SVM carregado sob demanda com sucesso.")
                
                if not models_loaded or embedder is None or human_svm is None:
                    err_msg = models_error if models_error else "Modelos de IA não carregados ou treinados."
                    response_data = {"error": err_msg}
                else:
                    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_img)
                    emb_res = embedder.embed(mp_image)
                    if emb_res and emb_res.embeddings:
                        emb = emb_res.embeddings[0].embedding
                        emb = np.array([emb], dtype=np.float32)
                        
                        _, pred = human_svm.predict(emb)
                        pred_class = int(pred[0][0])
                        
                        _, raw_margin = human_svm.predict(emb, flags=cv2.ml.StatModel_RAW_OUTPUT)
                        raw_val = float(raw_margin[0][0])
                        
                        confidence = 1.0 / (1.0 + np.exp(-abs(raw_val) * 2.0))
                        label = "Humano" if pred_class == 0 else "Não Humano"
                        
                        response_data = {
                            "prediction": label,
                            "confidence": round(confidence * 100, 1)
                        }
                    else:
                        response_data = {"error": "Não foi possível extrair os embeddings da face."}
                        
            except Exception as e:
                response_data = {"error": f"Erro interno do servidor: {str(e)}"}
                
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Access-Control-Allow-Methods', 'POST, GET, OPTIONS')
            self.send_header('Access-Control-Allow-Headers', 'X-Requested-With, Content-Type')
            self.end_headers()
            self.wfile.write(json.dumps(response_data).encode('utf-8'))
            
        else:
            super().do_POST()

def start_server():
    global PORT
    socketserver.TCPServer.allow_reuse_address = True
    try:
        with socketserver.TCPServer(("", PORT), Handler) as httpd:
            print(f"\n[+] Servidor local iniciado com sucesso!")
            print(f"    -> Diretório servido: {DIRECTORY}")
            print(f"    -> Endereço local: http://localhost:{PORT}/app_human.html")
            print("[*] Pressione Ctrl+C no console para encerrar o servidor.")
            httpd.serve_forever()
    except Exception as e:
        print(f"[!] Erro ao iniciar o servidor na porta {PORT}: {e}")
        print("[*] Tentando porta alternativa 8080...")
        try:
            PORT = 8080
            with socketserver.TCPServer(("", PORT), Handler) as httpd:
                print(f"\n[+] Servidor local iniciado com sucesso na porta {PORT}!")
                print(f"    -> Endereço local: http://localhost:{PORT}/app_human.html")
                httpd.serve_forever()
        except Exception as err:
            print(f"[!] Erro crítico: não foi possível iniciar o servidor. {err}")
            sys.exit(1)

def main():
    print("=" * 60)
    print("      INICIALIZADOR DA APLICACAO DE DETECCÃO HUMANA      ")
    print("=" * 60)
    
    model_loader = threading.Thread(target=init_models)
    model_loader.daemon = True
    model_loader.start()
    
    server_thread = threading.Thread(target=start_server)
    server_thread.daemon = True
    server_thread.start()
    
    time.sleep(1.0)
    
    app_url = f"http://localhost:{PORT}/app_human.html"
    print(f"\n[*] Abrindo a aplicação no seu navegador padrão...")
    print(f"    -> URL: {app_url}")
    
    try:
        webbrowser.open(app_url)
    except Exception as e:
        print(f"[!] Não foi possível abrir o navegador automaticamente: {e}")
        print(f"[!] Copie e cole este link no seu navegador: {app_url}")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\n[*] Encerrando o servidor local...")
        print("[+] Concluído. Tenha um excelente dia!")
        sys.exit(0)

if __name__ == "__main__":
    main()

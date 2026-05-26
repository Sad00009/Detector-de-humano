Detector de Humanos com Inteligência Artificial 
Descrição 
Sistema de detecção e classificação em tempo real utilizando Visão Computacional, 
MediaPipe, EfficientDet Lite, MobileNet V3 Embeddings e SVM. 
O projeto identifica automaticamente se o conteúdo detectado pertence à classe: - Humano - Não Humano 
com suporte para imagens, webcam em tempo real e API local de predição. 
Tecnologias Utilizadas - Python 3 - OpenCV - MediaPipe Tasks Vision - EfficientDet Lite0 - MobileNet V3 - SVM (Support Vector Machine) - HTML5 - CSS3 - JavaScript - WebAssembly 
Funcionalidades - Detecção de objetos em tempo real - Classificação Humano vs Não Humano - Upload de imagens - Suporte à webcam - Interface moderna e responsiva - API local /predict - Geração automática de imagens não humanas - Treinamento personalizado do classificador 
Estrutura do Projeto 
projeto/ 
├── app_human.html 
├── run_human_app.py 
├── train_human_classifier.py 
├── generate_non_humans.py 
├── models/ 
│   ├── efficientdet_lite0.tflite 
│   ├── mobilenet_v3_small.tflite 
│   └── human_svm.xml 
Como Funciona 
O sistema utiliza dois estágios de Inteligência Artificial. 
1. Detecção de Objetos: 
O modelo EfficientDet Lite0 detecta objetos presentes na imagem ou na webcam. 
2. Classificação Inteligente: 
Quando uma pessoa é detectada: - a região é recortada; - embeddings são extraídos utilizando MobileNet V3; - um classificador SVM determina se é Humano ou Não Humano. 
Instalação 
git clone https://github.com/seu-usuario/seu-repositorio.git 
cd seu-repositorio 
pip install opencv-python mediapipe numpy 
Treinamento do Modelo 
python train_human_classifier.py 
O modelo treinado será salvo em: 
models/human_svm.xml 
Gerando Dataset Não Humano 
python generate_non_humans.py 
O script gera imagens artificiais, baixa imagens aleatórias e constrói automaticamente a 
classe Não Humano. 
Executando a Aplicação 
python run_human_app.py 
Depois disso, abra: 
http://localhost:8000/app_human.html 
API Local 
Endpoint: 
POST /predict 
Exemplo JSON: 
{ 
"image": "base64_da_imagem" 
} 
Resposta: 
{ 
"prediction": "Humano", 
"confidence": 98.4 
} 
Objetivo Acadêmico 
Projeto desenvolvido para estudos de Inteligência Artificial, Machine Learning, Visão 
Computacional, Reconhecimento de Objetos e Classificação de Imagens. 
Possíveis Melhorias Futuras - reconhecimento facial - múltiplas classes - treinamento com Deep Learning - integração com YOLO - exportação para mobile - deploy online 
Autor 
Gabriel Sad 
Licença 
Este projeto é destinado para fins educacionais e acadêmicos.

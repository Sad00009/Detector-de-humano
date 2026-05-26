# -*- coding: utf-8 -*-
"""
Script to train a custom human vs non-human classifier (SVM) using MediaPipe MobileNet V3 Embeddings.
"""
import os
import sys
import glob
import cv2
import numpy as np

# Prevent shadowing the installed mediapipe library
if '' in sys.path:
    sys.path.remove('')
if os.path.dirname(os.path.abspath(__file__)) in sys.path:
    sys.path.remove(os.path.dirname(os.path.abspath(__file__)))

import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python import BaseOptions

def load_embeddings(embedder, class_dir, label):
    embeddings = []
    labels = []
    
    if not os.path.exists(class_dir):
        print(f"[!] Diretório {class_dir} não existe.")
        return embeddings, labels
        
    extensions = ('*.jpg', '*.jpeg', '*.png')
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(class_dir, ext)))
        
    print(f"[*] Processando {len(image_paths)} imagens na pasta {os.path.basename(class_dir)}...")
    
    count = 0
    for img_path in image_paths:
        try:
            img = cv2.imread(img_path)
            if img is None:
                continue
                
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_img)
            embedding_result = embedder.embed(mp_image)
            
            if embedding_result and embedding_result.embeddings:
                emb_vector = embedding_result.embeddings[0].embedding
                if len(emb_vector) == 1024:
                    embeddings.append(emb_vector)
                    labels.append(label)
                    count += 1
            
            if count % 200 == 0 and count > 0:
                print(f"    -> {count} imagens processadas com sucesso...")
                
        except Exception as e:
            continue
            
    print(f"[+] Total de embeddings extraídos com sucesso para a classe {label} ({os.path.basename(class_dir)}): {count}/{len(image_paths)}")
    return embeddings, labels

def main():
    print("=" * 60)
    print("   TREINAMENTO DO CLASSIFICADOR HUMANO VS NÃO HUMANO    ")
    print("=" * 60)
    
    model_path = "D:/DOWNLOADS/mediapipe-master/models/mobilenet_v3_small.tflite"
    dataset_dir = "D:/DOWNLOADS/gender_dataset"
    if not os.path.exists(dataset_dir):
        dataset_dir = "D:/DOWNLOADS/archive/gender_dataset"
        
    svm_save_path = "D:/DOWNLOADS/mediapipe-master/models/human_svm.xml"
    
    if not os.path.exists(model_path):
        print(f"[!] Erro: Modelo base não encontrado em {model_path}")
        sys.exit(1)
        
    if not os.path.exists(dataset_dir):
        print(f"[!] Erro: Dataset não encontrado em {dataset_dir}")
        sys.exit(1)
        
    print("[*] Inicializando MediaPipe Image Embedder...")
    try:
        options = vision.ImageEmbedderOptions(
            base_options=BaseOptions(model_asset_path=model_path)
        )
        embedder = vision.ImageEmbedder.create_from_options(options)
    except Exception as e:
        print(f"[!] Falha ao inicializar o embedder: {e}")
        sys.exit(1)
        
    men_dir = os.path.join(dataset_dir, "men")
    women_dir = os.path.join(dataset_dir, "women")
    non_human_dir = os.path.join(dataset_dir, "non_human")
    
    print("[*] Extraindo características da classe: Humano (Men)")
    men_embs, men_lbls = load_embeddings(embedder, men_dir, 0)
    
    print("[*] Extraindo características da classe: Humano (Women)")
    women_embs, women_lbls = load_embeddings(embedder, women_dir, 0)
    
    print("[*] Extraindo características da classe: Não Humano (Non-Human)")
    non_human_embs, non_human_lbls = load_embeddings(embedder, non_human_dir, 1)
    
    if not (men_embs or women_embs) or not non_human_embs:
        print("[!] Erro: Não foi possível extrair embeddings suficientes.")
        sys.exit(1)
        
    X = np.array(men_embs + women_embs + non_human_embs, dtype=np.float32)
    Y = np.array(men_lbls + women_lbls + non_human_lbls, dtype=np.int32)
    
    print(f"\n[*] Dataset Total: {X.shape[0]} amostras com {X.shape[1]} características cada.")
    
    indices = np.arange(X.shape[0])
    np.random.seed(42)
    np.random.shuffle(indices)
    
    X_shuffled = X[indices]
    Y_shuffled = Y[indices]
    
    split_idx = int(0.8 * X.shape[0])
    X_train, X_test = X_shuffled[:split_idx], X_shuffled[split_idx:]
    Y_train, Y_test = Y_shuffled[:split_idx], Y_shuffled[split_idx:]
    
    print(f"[*] Treino: {X_train.shape[0]} amostras, Teste: {X_test.shape[0]} amostras.")
    
    print("\n[*] Treinando classificador SVM (Kernel Linear)...")
    try:
        svm = cv2.ml.SVM_create()
        svm.setType(cv2.ml.SVM_C_SVC)
        svm.setKernel(cv2.ml.SVM_LINEAR)
        svm.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER, 500, 1e-6))
        
        svm.train(X_train, cv2.ml.ROW_SAMPLE, Y_train)
        print("[+] Treinamento concluído com sucesso!")
        
        print("\n[*] Avaliando precisão do classificador...")
        
        _, pred_train = svm.predict(X_train)
        pred_train = pred_train.flatten().astype(np.int32)
        train_acc = np.mean(pred_train == Y_train) * 100
        
        _, pred_test = svm.predict(X_test)
        pred_test = pred_test.flatten().astype(np.int32)
        test_acc = np.mean(pred_test == Y_test) * 100
        
        print(f"    -> Acurácia no Conjunto de Treino: {train_acc:.2f}%")
        print(f"    -> Acurácia no Conjunto de Teste:  {test_acc:.2f}%")
        
        os.makedirs(os.path.dirname(svm_save_path), exist_ok=True)
        svm.save(svm_save_path)
        print(f"\n[+] Modelo SVM de Detecção Humana salvo com sucesso em: {svm_save_path}")
        print("=" * 60)
        
    except Exception as e:
        print(f"[!] Erro ao treinar/salvar o modelo SVM: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

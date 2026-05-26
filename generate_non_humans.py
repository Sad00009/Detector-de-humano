# -*- coding: utf-8 -*-
"""
Script to generate synthetic and download real non-human images to D:\\DOWNLOADS\\gender_dataset\\non_human.
Created to construct the 'Não Humano' class for classifier training.
"""
import os
import cv2
import numpy as np
import urllib.request
import random
import time
import socket

def generate_synthetic_images(output_dir, count=400):
    print(f"[*] Gerando {count} imagens sintéticas não-humanas em {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    
    for i in range(count):
        h, w = 224, 224
        img_type = i % 5
        
        if img_type == 0:
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            img = np.full((h, w, 3), color, dtype=np.uint8)
            
        elif img_type == 1:
            noise_type = random.choice(['uniform', 'gaussian'])
            if noise_type == 'uniform':
                img = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
            else:
                mean = 127
                stddev = 40
                gaussian_noise = np.random.normal(mean, stddev, (h, w, 3))
                img = np.clip(gaussian_noise, 0, 255).astype(np.uint8)
                
        elif img_type == 2:
            img = np.full((h, w, 3), (random.randint(200, 255), random.randint(200, 255), random.randint(200, 255)), dtype=np.uint8)
            for _ in range(random.randint(5, 15)):
                shape = random.choice(['circle', 'rect', 'line'])
                color = (random.randint(0, 180), random.randint(0, 180), random.randint(0, 180))
                thickness = random.randint(1, 5)
                
                if shape == 'circle':
                    center = (random.randint(0, w), random.randint(0, h))
                    radius = random.randint(10, w // 4)
                    cv2.circle(img, center, radius, color, thickness)
                elif shape == 'rect':
                    pt1 = (random.randint(0, w), random.randint(0, h))
                    pt2 = (random.randint(0, w), random.randint(0, h))
                    cv2.rectangle(img, pt1, pt2, color, thickness)
                else:
                    pt1 = (random.randint(0, w), random.randint(0, h))
                    pt2 = (random.randint(0, w), random.randint(0, h))
                    cv2.line(img, pt1, pt2, color, thickness)
                    
        elif img_type == 3:
            img = np.zeros((h, w, 3), dtype=np.uint8)
            c1 = np.array([random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)])
            c2 = np.array([random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)])
            
            for y in range(h):
                alpha = y / float(h)
                color = c1 * (1 - alpha) + c2 * alpha
                img[y, :, :] = color.astype(np.uint8)
                
        else:
            img = np.full((h, w, 3), (255, 255, 255), dtype=np.uint8)
            grid_size = random.randint(10, 40)
            color = (random.randint(0, 150), random.randint(0, 150), random.randint(0, 150))
            for x in range(0, w, grid_size):
                cv2.line(img, (x, 0), (x, h), color, 1)
            for y in range(0, h, grid_size):
                cv2.line(img, (0, y), (w, y), color, 1)
                
        cv2.imwrite(os.path.join(output_dir, f"synthetic_{i:04d}.png"), img)
        
    print(f"[+] Concluído! Geradas {count} imagens sintéticas em {output_dir}")

def download_picsum_images(output_dir, count=400):
    print(f"\n[*] Tentando baixar {count} imagens reais de objetos/natureza via Picsum Photos...")
    os.makedirs(output_dir, exist_ok=True)
    
    url = "https://picsum.photos/224/224"
    success_count = 0
    socket.setdefaulttimeout(5.0)
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    
    for i in range(count):
        img_path = os.path.join(output_dir, f"picsum_{i:04d}.jpg")
        
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req) as response:
                image_data = response.read()
                
            nparr = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is not None:
                cv2.imwrite(img_path, img)
                success_count += 1
                if success_count % 50 == 0:
                    print(f"    -> Baixadas {success_count}/{count} imagens com sucesso...")
            else:
                print(f"    [!] Erro ao decodificar imagem {i}, pulando...")
                
            time.sleep(0.1)
            
        except Exception as e:
            print(f"    [!] Falha na conexão ou erro no download: {e}")
            print("    [!] Interrompendo downloads e prosseguindo apenas com as imagens sintéticas/existentes.")
            break
            
    print(f"[+] Concluído! Total de imagens baixadas com sucesso: {success_count}/{count}")
    return success_count

def main():
    print("=" * 60)
    dirs = [
        "D:/DOWNLOADS/gender_dataset/non_human",
        "D:/DOWNLOADS/archive/gender_dataset/non_human"
    ]
    
    for directory in dirs:
        print(f"[*] Configurando diretório: {directory}")
        os.makedirs(directory, exist_ok=True)
        generate_synthetic_images(directory, count=400)
        download_picsum_images(directory, count=400)
        
    print("\n[+] Configuração do dataset 'Não Humano' concluída com sucesso!")
    print("=" * 60)

if __name__ == "__main__":
    main()

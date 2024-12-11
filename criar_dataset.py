import os  # Biblioteca para manipulação de diretórios e arquivos
import pickle  # Biblioteca para serialização e salvamento de dados

import mediapipe as mp  # Biblioteca para detecção e rastreamento de landmarks
import cv2  # OpenCV, para carregamento e manipulação de imagens
import matplotlib.pyplot as plt  # Biblioteca para visualização (não utilizada aqui)

# Configurações do MediaPipe para detecção de mãos
mp_hands = mp.solutions.hands  # Módulo de detecção de mãos
mp_drawing = mp.solutions.drawing_utils  # Utilitário para desenhar landmarks
mp_drawing_styles = mp.solutions.drawing_styles  # Estilos para visualização dos desenhos

# Inicialização do modelo de detecção de mãos
hands = mp_hands.Hands(
    static_image_mode=True,  # Modo para processar imagens estáticas
    min_detection_confidence=0.3  # Confiança mínima para considerar uma detecção válida
)

# Diretório onde estão os dados organizados em subpastas por classe/rótulo
DATA_DIR = './data'

# Listas para armazenar os vetores de características (data) e os rótulos (labels)
data = []  # Lista de características extraídas
labels = []  # Lista de rótulos correspondentes

# Itera sobre cada subpasta (rótulo) no diretório de dados
for dir_ in os.listdir(DATA_DIR):
    # Itera sobre cada imagem na subpasta
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        data_aux = []  # Lista auxiliar para armazenar as características de uma imagem
        x_ = []  # Lista para armazenar as coordenadas x de landmarks
        y_ = []  # Lista para armazenar as coordenadas y de landmarks

        # Carrega a imagem em BGR e converte para RGB
        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))  # Carrega a imagem
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Converte a imagem para RGB, necessário para o MediaPipe.


        # Processa a imagem com o modelo de detecção de mãos
        results = hands.process(img_rgb)  # Obtém os landmarks da mão

        # Verifica se foram detectados landmarks na imagem
        if results.multi_hand_landmarks:
            # Itera sobre cada conjunto de landmarks detectados
            for hand_landmarks in results.multi_hand_landmarks:
                # Extrai as coordenadas x e y de cada ponto (landmark)
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x  # Coordenada x normalizada
                    y = hand_landmarks.landmark[i].y  # Coordenada y normalizada
                    x_.append(x)  # Adiciona x à lista de coordenadas x
                    y_.append(y)  # Adiciona y à lista de coordenadas y

                # Normaliza os pontos em relação ao mínimo (para centralizar no ponto mínimo)
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x  # Coordenada x
                    y = hand_landmarks.landmark[i].y  # Coordenada y
                    data_aux.append(x - min(x_))  # Normaliza x subtraindo o valor mínimo de x_
                    data_aux.append(y - min(y_))  # Normaliza y subtraindo o valor mínimo de y_

            # Adiciona as características da imagem à lista principal
            data.append(data_aux)  # Adiciona o vetor de características da imagem
            labels.append(dir_)  # Adiciona o rótulo correspondente

# Salva os dados e rótulos em um arquivo binário usando pickle
f = open('data.pickle', 'wb')  # Abre o arquivo para escrita binária
pickle.dump({'data': data, 'labels': labels}, f)  # Serializa os dados e rótulos
f.close()  # Fecha o arquivo

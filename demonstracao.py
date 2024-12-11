import os  # Biblioteca para manipulação de diretórios e arquivos

import mediapipe as mp  # Biblioteca para detecção e rastreamento de landmarks
import cv2  # OpenCV, usada para carregar e manipular imagens
import matplotlib.pyplot as plt  # Biblioteca para visualização de imagens

# Configurações do MediaPipe para detecção de mãos
mp_hands = mp.solutions.hands  # Módulo de detecção de mãos
mp_drawing = mp.solutions.drawing_utils  # Utilitário para desenhar landmarks
mp_drawing_styles = mp.solutions.drawing_styles  # Estilos para visualização dos desenhos

# Inicialização do modelo de detecção de mãos do MediaPipe
hands = mp_hands.Hands(
    static_image_mode=True,  # Define que o modelo processará imagens estáticas
    min_detection_confidence=0.3  # Define a confiança mínima para considerar uma detecção válida
)

# Diretório onde as imagens estão organizadas em subpastas por classe/rótulo
DATA_DIR = './data'

# Itera sobre cada subpasta (rótulo) no diretório de dados
for dir_ in os.listdir(DATA_DIR):
    # Itera sobre a primeira imagem de cada subpasta ([:1] pega apenas o primeiro arquivo)
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_))[:1]:
        # Carrega a imagem no formato BGR usando OpenCV
        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        # Converte a imagem para o formato RGB (necessário para o MediaPipe)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Processa a imagem com o modelo MediaPipe para detectar landmarks da mão
        results = hands.process(img_rgb)

        # Verifica se algum conjunto de landmarks foi detectado na imagem
        if results.multi_hand_landmarks:
            # Itera sobre cada conjunto de landmarks detectados (suporta múltiplas mãos)
            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    print(hand_landmarks.landmark[i])
               

        # Cria uma nova figura para exibir a imagem com landmarks desenhados
        plt.figure()
        plt.imshow(img_rgb)  # Exibe a imagem em formato RGB

# Exibe todas as imagens desenhadas em sequência
plt.show()

import pickle
import cv2
import mediapipe as mp
import numpy as np

# Abre o arquivo 'model.p' no modo de leitura binária e carrega o modelo salvo.
model_dict = pickle.load(open('./model.p', 'rb'))

# Extrai o modelo treinado do dicionário carregado.
model = model_dict['model']

# Inicializa a captura de vídeo usando a câmera padrão do sistema.
cap = cv2.VideoCapture(0)

# Importa a solução de detecção de mãos do MediaPipe.
mp_hands = mp.solutions.hands

# Importa as ferramentas de desenho do MediaPipe para visualizar landmarks.
mp_drawing = mp.solutions.drawing_utils

# Importa estilos predefinidos para desenhar landmarks e conexões.
mp_drawing_styles = mp.solutions.drawing_styles

# Inicializa o detector de mãos, configurado para trabalhar com imagens estáticas e uma confiança mínima de 30% para a detecção.
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Define um dicionário para mapear os índices das classes preditas pelo modelo para os respectivos caracteres.
labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
               10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T',
               20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z', 26: '0', 27: '1', 28: '2', 29: '3',
               30: '4', 31: '5', 32: '6', 33: '7', 34: '8', 35: '9'}

# Inicia um loop infinito para processar os frames da câmera.
while True:
    # Lista auxiliar para armazenar as características (features) normalizadas.
    data_aux = []

    # Lista para armazenar as coordenadas x dos landmarks detectados.
    x_ = []

    # Lista para armazenar as coordenadas y dos landmarks detectados.
    y_ = []

    # Captura um frame da câmera. 'ret' indica se a captura foi bem-sucedida.
    ret, frame = cap.read()

    # Verifica se a captura falhou; em caso positivo, interrompe o loop.
    if not ret:
        break

    # Obtém a altura (H), largura (W) e número de canais (não usado) do frame.
    H, W, _ = frame.shape

    # Converte o frame de BGR (formato padrão do OpenCV) para RGB (necessário pelo MediaPipe).
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Processa o frame usando o detector de mãos do MediaPipe para obter landmarks.
    results = hands.process(frame_rgb)

    # Verifica se foram detectadas mãos no frame.
    if results.multi_hand_landmarks:
        # Itera sobre as landmarks de cada mão detectada.
        for hand_landmarks in results.multi_hand_landmarks:
            # Desenha as landmarks e conexões no frame.
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

        # Itera novamente sobre as landmarks para extrair suas coordenadas.
        for hand_landmarks in results.multi_hand_landmarks:
            # Itera sobre cada ponto das landmarks.
            for i in range(len(hand_landmarks.landmark)):
                # Obtém a coordenada x normalizada do ponto.
                x = hand_landmarks.landmark[i].x

                # Obtém a coordenada y normalizada do ponto.
                y = hand_landmarks.landmark[i].y

                # Armazena a coordenada x.
                x_.append(x)

                # Armazena a coordenada y.
                y_.append(y)

            # Itera novamente para normalizar as coordenadas.
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y

                # Normaliza a coordenada x subtraindo o valor mínimo.
                data_aux.append(x - min(x_))

                # Normaliza a coordenada y subtraindo o valor mínimo.
                data_aux.append(y - min(y_))

        # Verifica o tamanho do vetor de características.
        if len(data_aux) == 42:
            # Se o vetor de características tiver o tamanho correto (42), faz a previsão.
            prediction = model.predict([np.asarray(data_aux)])

            # Obtém o caractere correspondente ao índice predito.
            predicted_character = labels_dict[int(prediction[0])]

            # Desenha um retângulo ao redor da mão detectada.
            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10
            x2 = int(max(x_) * W) - 10
            y2 = int(max(y_) * H) - 10

            # Desenha o retângulo no frame.
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)

            # Adiciona o caractere predito acima do retângulo.
            cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                        1.3, (0, 0, 0), 3, cv2.LINE_AA)

        else:
            # Se o vetor de características não tiver o tamanho correto, preenche com zeros.
            data_aux = [0] * 42
            prediction = model.predict([np.asarray(data_aux)])
            predicted_character = labels_dict[int(prediction[0])]

    # Exibe a imagem com os resultados.
    cv2.imshow('frame', frame)

    # Verifica se a tecla 'q' foi pressionada para sair do loop.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera o recurso de captura da câmera.
cap.release()

# Fecha todas as janelas criadas pelo OpenCV.
cv2.destroyAllWindows()

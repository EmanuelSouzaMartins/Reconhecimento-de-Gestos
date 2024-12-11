import pickle
# Importa o modelo RandomForestClassifier da biblioteca scikit-learn.
from sklearn.ensemble import RandomForestClassifier
# Importa a função para dividir o conjunto de dados em treino e teste.
from sklearn.model_selection import train_test_split
# Importa a função para calcular a acurácia do modelo.
from sklearn.metrics import accuracy_score
import numpy as np

# Carrega os dados armazenados no arquivo data.pickle usando pickle.
data_dict = pickle.load(open('./data.pickle', 'rb'))

# Extrai os dados e os rótulos do dicionário e os converte em arrays numpy.
data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# Divide os dados em conjuntos de treino e teste.
# Usa 80% dos dados para treino e 20% para teste.
# O parâmetro shuffle embaralha os dados, e stratify preserva a proporção das classes.
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Instancia o modelo de classificação Random Forest.
model = RandomForestClassifier()

# Treina o modelo usando os dados e rótulos do conjunto de treino.
model.fit(x_train, y_train)

# Faz previsões no conjunto de teste.
y_predict = model.predict(x_test)

# Calcula a acurácia das previsões comparando com os rótulos reais.
score = accuracy_score(y_predict, y_test)

# Exibe a porcentagem de amostras classificadas corretamente.
print('{}% of samples were classified correctly !'.format(score * 100))

# Abre um arquivo para salvar o modelo treinado no formato binário.
f = open('model.p', 'wb')

# Serializa o modelo treinado e o salva em um arquivo.
pickle.dump({'model': model}, f)

# Fecha o arquivo para concluir a escrita.
f.close()

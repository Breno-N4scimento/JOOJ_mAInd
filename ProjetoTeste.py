import unittest
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Função auxiliar para carregar e preparar os dados
def preparar_dados(caminho_csv):
    data = pd.read_csv(caminho_csv)
    convertVar = LabelEncoder()

    for column in data.columns[:-1]:
        data[column] = convertVar.fit_transform(data[column])
    data['Disorder'] = convertVar.fit_transform(data['Disorder'])

    X = data.drop(columns=['Disorder'])
    Y = data['Disorder']
    return train_test_split(X, Y, test_size=0.2, random_state=42)


def inicializarModelo():
    return KNeighborsClassifier()

class TestChatbotModel(unittest.TestCase):

    #teste de carregamento de dataset
    def testCarregamentoDataset(self):
        try:
            data = pd.read_csv("dataset.csv")
        except FileNotFoundError:
            self.fail("Arquivo dataset.csv não foi encontrado")

    # teste de ausência do dataset
    def testAusenciaDataset(self):
        with self.assertRaises(FileNotFoundError):
            pd.read_csv("arquivo_inexistente.csv")

    #teste a conversão das variáveis
    def testConversaoVar(self):
        data = pd.read_csv("dataset.csv")
        convertVar = LabelEncoder()

        for column in data.columns[:-1]:
            data[column] = convertVar.fit_transform(data[column])
            self.assertIn(data[column].dtype, ['int32', 'int64'])

    #testa a divisão dos dados
    def testDivDados(self):
        xTreino, xTeste, yTreino, yTeste = preparar_dados("dataset.csv")
        self.assertGreater(len(xTreino), 0)
        self.assertGreater(len(xTeste), 0)
        self.assertGreater(len(yTreino), 0)
        self.assertGreater(len(yTeste), 0)

    # teste de inicialização do modelo KNN
    def testInicializacaoKnn(self):
        modelo = inicializarModelo()
        self.assertIsInstance(modelo, KNeighborsClassifier)
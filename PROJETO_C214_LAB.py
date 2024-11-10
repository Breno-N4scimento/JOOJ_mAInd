import pandas as pd
import skfuzzy
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Carregar os dados CSV
data = pd.read_csv(r"dataset.csv")

# Converter valores categóricos em valores numéricos (usado no KNN)
convertVar = LabelEncoder()

# Aplica LabelEncoder em todas as colunas categóricas de X
for column in data.columns[:-1]:  # Exclui a última coluna 'Disorder'
    data[column] = convertVar.fit_transform(data[column])

# Codificar a variável target (Disorder)
data['Disorder'] = convertVar.fit_transform(data['Disorder'])

# Dividir os dados em features (X) e rótulos (y)
X = data.drop(columns=['Disorder'])  # Todas as colunas, exceto a última (entrada)
Y = data['Disorder']  # A última coluna (saída)

# Divide os dados em conjunto de treino e teste
xTreino, xTeste, yTreino, yTeste = train_test_split(X, Y, test_size=0.2, random_state=42)

# Inicializar o modelo
modelo = KNeighborsClassifier()

# Treinar o modelo
modelo.fit(xTreino, yTreino)

# Perguntas baseadas nas colunas do dataset
conversas = [
    "Você está se sentindo nervoso?",
    "Você está tendo ataques de pânico?",
    "Sua respiração está rápida?",
    "Você está suando?",
    "Está tendo problemas para se concentrar?",
    "Está tendo dificuldades para dormir?",
    "Está tendo problemas no trabalho?",
    "Você se sente sem esperança?",
    "Você está com raiva?",
    "Você tende a exagerar?",
    "Você percebe mudanças nos seus hábitos alimentares?",
    "Você tem pensamentos suicidas?",
    "Você se sente cansado?",
    "Você tem um amigo próximo?",
    "Você tem vício em redes sociais?",
    "Você ganhou peso recentemente?",
    "Você valoriza muito as posses materiais?",
    "Você se considera introvertido?",
    "Lembranças estressantes estão surgindo?",
    "Você tem pesadelos?",
    "Você evita pessoas ou atividades?",
    "Você está se sentindo negativo?",
    "Está com problemas de concentração?",
    "Você tende a se culpar por coisas?"
]

#dicionário com graus de pertinência para "não sei"
graus_perc = {
    "Você está se sentindo nervoso?": 0.5,
    "Você está tendo ataques de pânico?": 0.7,
    "Sua respiração está rápida?": 0.6,
    "Você está suando?": 0.5,
    "Está tendo problemas para se concentrar?": 0.5,
    "Está tendo dificuldades para dormir?": 0.6,
    "Está tendo problemas no trabalho?": 0.5,
    "Você se sente sem esperança?": 0.8,
    "Você está com raiva?": 0.4,
    "Você tende a exagerar?": 0.3,
    "Você percebe mudanças nos seus hábitos alimentares?": 0.5,
    "Você tem pensamentos suicidas?": 0.9,
    "Você se sente cansado?": 0.6,
    "Você tem um amigo próximo?": 0.3,
    "Você tem vício em redes sociais?": 0.4,
    "Você ganhou peso recentemente?": 0.5,
    "Você valoriza muito as posses materiais?": 0.4,
    "Você se considera introvertido?": 0.4,
    "Lembranças estressantes estão surgindo?": 0.6,
    "Você tem pesadelos?": 0.5,
    "Você evita pessoas ou atividades?": 0.7,
    "Você está se sentindo negativo?": 0.7,
    "Está com problemas de concentração?": 0.5,
    "Você tende a se culpar por coisas?": 0.6,
}

#coletar respostas do usuário e fazer a predição
def coletarRespostas(nome_usuario):
    respostas_usuario = []

    for pergunta in conversas:
        while True:
            resposta = input("{} (sim/não/não sei): ".format(pergunta)).strip().lower()
            if resposta == 's':
                respostas_usuario.append(1.0)
                break
            elif resposta == 'n':
                respostas_usuario.append(0.0)
                break
            elif resposta == 'ns':
                if pergunta in graus_perc:
                    respostas_usuario.append(graus_perc[pergunta])
                else:
                    respostas_usuario.append(0.5)
                break
            else:
                print("Por favor, insira 'sim', 'não' ou 'não sei'.")

    #fazer a predição com base nas respostas do usuário
    predicao = modelo.predict([respostas_usuario])

    #converter a predição de volta para o transtorno correspondente
    transtornoPredito = convertVar.inverse_transform(predicao)[0]

    #diagnósticos
    if transtornoPredito == 'Normal':
        print("Com base nas suas respostas, você pode estar suave (normal).")
    elif transtornoPredito == 'Stress':
        print("Com base nas informações dadas, você pode estar com estresse.")
    elif transtornoPredito == 'Loneliness':
        print("Com base nas informações, você pode estar se sentindo sozinho.")
    elif transtornoPredito == 'Depression':
        print("Com base nas informações que você forneceu, você indica sintomas de depressão.")
    elif transtornoPredito == 'Anxiety':
        print("Com base nas informações que você forneceu, parece que você está com ansiedade.")

    #salvar as respostas do usuário em um datarame
    user_data = pd.DataFrame([respostas_usuario], columns=conversas)
    user_data.insert(0, 'Nome', nome_usuario)  # Adicionar o nome do usuário na primeira coluna
    user_data['Predição'] = transtornoPredito

    #salvar infos no CSV
    try:
        user_data.to_csv('respostas_usuarios.csv', mode='a', header=False, index=False)
    except Exception as e:
        print("Erro ao salvar as respostas no CSV: ", e)


#funcao para buscar um usuário pelo nome
def buscarUsuario(nome):
    try:
        #carregar o arquivo com as respostas dos usuários
        df = pd.read_csv('respostas_usuarios.csv', header=None, names=['Nome'] + conversas + ['Predição'])
        resultado = df[df['Nome'] == nome]
        if not resultado.empty:
            print("Diagnóstico para {}: {}".format(nome, resultado.iloc[0]['Predição']))
        else:
            print("Nenhum usuário encontrado com esse nome.")
    except FileNotFoundError:
        print("Arquivo de respostas não encontrado.")


#funcao para inserir novo usuário ou pesquisar
def iniciarSistema():
    while True:
        print("\nEscolha uma opção:")
        print("1 - Inserir novo usuário")
        print("2 - Pesquisar usuário")
        print("3 - Sair")

        opcao = input("Digite a opção desejada: ").strip()

        if opcao == '1':
            nome = input("Digite seu nome: ").strip()
            coletarRespostas(nome)
        elif opcao == '2':
            nome = input("Digite o nome do usuário para pesquisa: ").strip()
            buscarUsuario(nome)
        elif opcao == '3':
            print("Saindo do sistema.")
            break
        else:
            print("Opção inválida. Tente novamente.")


#iniciar o sistema
iniciarSistema()

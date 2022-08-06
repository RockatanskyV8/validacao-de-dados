import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from sklearn.model_selection import KFold
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import cross_validate
from sklearn.model_selection import StratifiedKFold

URI="https://gist.githubusercontent.com/guilhermesilveira/e99a526b2e7ccc6c3b70f53db43a87d2/raw/1605fc74aa778066bf2e6695e24d53cf65f2f447/machine-learning-carros-simulacao.csv"

dados = pd.read_csv(URI).drop(columns=["Unnamed: 0"], axis=1)
#  print(dados.head())

x = dados[["preco", "idade_do_modelo", "km_por_ano"]]
y = dados["vendido"]

SEED = 158020
np.random.seed(SEED)

treino_x, teste_x, treino_y, teste_y = train_test_split(x, y, test_size = 0.25, stratify = y)
#  print( "Treinaremos com %d elementos e testaremos com %d elementos" % (len(treino_x), len(teste_x)) )


dummy_stratified = DummyClassifier()
dummy_stratified.fit(treino_x, treino_y)
acuracia = dummy_stratified.score(teste_x, teste_y) * 100

#  print( "A acurácia do dummy_stratified foi %.2f%%" % acuracia )

modelo = DecisionTreeClassifier(max_depth=2)
modelo.fit(treino_x, treino_y)
previsoes = modelo.predict(teste_x)

acuracia = accuracy_score(teste_y, previsoes) * 100
#  print("A acurácia foi %.2f%%" % acuracia)


def acuracia_decision_tree(CV):
    modelo = DecisionTreeClassifier(max_depth=2)
    results = cross_validate(modelo, x, y, cv = CV, return_train_score=False)
    media = results['test_score'].mean()
    desvio_padrao = results['test_score'].std()
    print("Accuracy com cross validation, %d = [%.2f, %.2f]" % (CV, (media - 2 * desvio_padrao)*100, (media + 2 * desvio_padrao) * 100))

def imprime_resultados(results):
    media = results['test_score'].mean()
    desvio_padrao = results['test_score'].std()
    print(" =================================== inicio =================================== ")
    print("Accuracy médio: %.2f" % (media * 100))
    print("Accuracy intervalo: [%.2f, %.2f]" % ((media - 2 * desvio_padrao)*100, (media + 2 * desvio_padrao) * 100))
    print(" ===================================  fim   =================================== ")

def kfold_decision_tree(**kwargs):
    cv = KFold(**kwargs)
    modelo = DecisionTreeClassifier(max_depth=2)
    results = cross_validate(modelo, x, y, cv = cv, return_train_score=False)
    imprime_resultados(results)

def stratified_kfold_decision_tree(dados_azar, **kwargs):
    x_azar = dados_azar[["preco", "idade_do_modelo","km_por_ano"]]
    y_azar = dados_azar["vendido"]
    cv = StratifiedKFold(**kwargs)
    modelo = DecisionTreeClassifier(max_depth=2)
    results = cross_validate(modelo, x, y, cv = cv, return_train_score=False)
    imprime_resultados(results)

def group_kfold_decision_tree(dados_azar, **kwargs):
    x_azar = dados_azar[["preco", "idade_do_modelo","km_por_ano"]]
    y_azar = dados_azar["vendido"]
    cv = GroupKFold(**kwargs)
    modelo = DecisionTreeClassifier(max_depth=2)
    results = cross_validate(modelo, x_azar, y_azar, cv = cv, groups = dados.modelo, return_train_score=False)
    imprime_resultados(results)

def pipeline_svn (dados):
    x = dados[["preco", "idade_do_modelo","km_por_ano"]]
    y = dados["vendido"]
    
    treino_x, teste_x, treino_y, teste_y = train_test_split(x, y, test_size = 0.25, stratify = y)
    scaler = StandardScaler()
    scaler.fit(treino_x)
    treino_x_escalado = scaler.transform(treino_x)
    teste_x_escalado = scaler.transform(teste_x)

    modelo = SVC()
    pipeline = Pipeline([('transformacao',scaler), ('estimador',modelo)])
    cv = GroupKFold(n_splits = 10)
    results = cross_validate(pipeline, treino_x_escalado, y_azar, cv = cv, groups = dados.modelo, return_train_score=False)
    imprime_resultados(results)

def pipeline_svn (dados_azar):
    x_azar = dados_azar[["preco", "idade_do_modelo","km_por_ano"]]
    y_azar = dados_azar["vendido"]
    scaler = StandardScaler()
    modelo = SVC()
    pipeline = Pipeline([('transformacao',scaler), ('estimador',modelo)])
    cv = GroupKFold(n_splits = 10)
    results = cross_validate(pipeline, x_azar, y_azar, cv = cv, groups = dados.modelo, return_train_score=False)
    imprime_resultados(results)

SEED = 301
np.random.seed(SEED)
#  acuracia_decision_tree(3)
#  acuracia_decision_tree(10)
#  acuracia_decision_tree(5)


#  kfold_decision_tree(n_splits=2)
#  kfold_decision_tree(n_splits=10, shuffle=True)


np.random.seed(SEED)
dados['modelo'] = dados.idade_do_modelo + np.random.randint(-2, 3, size=10000)
dados.modelo = dados.modelo + abs(dados.modelo.min()) + 1
#  print(dados.head())
dados_azar = dados.sort_values("vendido", ascending=True)
#  print(dados_azar.head())

stratified_kfold_decision_tree(dados_azar, n_splits = 10, shuffle=True)
#  group_kfold_decision_tree(dados_azar, n_splits = 10)
#  pipeline_svn(dados_azar)


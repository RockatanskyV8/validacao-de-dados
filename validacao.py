import pandas as pd
import numpy as np

from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from sklearn.model_selection import KFold
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import StratifiedKFold

from modelos import Split 
from modelos import Modelos 

SEED = 301
np.random.seed(SEED)

URI="https://gist.githubusercontent.com/guilhermesilveira/e99a526b2e7ccc6c3b70f53db43a87d2/raw/1605fc74aa778066bf2e6695e24d53cf65f2f447/machine-learning-carros-simulacao.csv"

df = pd.read_csv(URI).drop(columns=["Unnamed: 0"], axis=1)
df['modelo'] = df.idade_do_modelo + np.random.randint(-2, 3, size=10000)
df.modelo = df.modelo + abs(df.modelo.min()) + 1

x = df[["preco", "idade_do_modelo", "km_por_ano"]]
y = df["vendido"]

#  dados = Split(158020, x, y)
modelos = Modelos(158020, x, y)


cvKFold           = KFold(n_splits = 10)
#  cvStratifiedKFold = StratifiedKFold(**kwargs)
#  cvGroupKFold      = GroupKFold(**kwargs)

arvore_decisao_2d = modelos.arvore_decisao_2d(cvKFold, df.modelo, 2, 128)
arvore_decisao_3d = modelos.arvore_decisao_3d(cvKFold, df.modelo, 2, 128, 128)

print(arvore_decisao_2d)
print(arvore_decisao_3d)


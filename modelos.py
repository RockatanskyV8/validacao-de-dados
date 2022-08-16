from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
import numpy as np
import pandas as pd

class Split:

    def __init__(self, seed, x, y):
       self.SEED = seed
       self.x = x
       self.y = y

    def test_split(self):
       np.random.seed(self.SEED)
       return train_test_split(self.x, self.y, test_size = 0.25, stratify = self.y)


class Modelos:
    
    def __init__(self, seed, x, y):
       self.SEED = seed
       self.x = x
       self.y = y

    def arvore_decisao(self, CV, GROUPS, MAX_DEPTH, MIN_SAMPLES_LEAF):
       np.random.seed(self.SEED)
       modelo = DecisionTreeClassifier(max_depth=MAX_DEPTH, min_samples_leaf=MIN_SAMPLES_LEAF)
       results = cross_validate(modelo, self.x, self.y, cv = CV, groups = GROUPS, return_train_score=True)
       train_score = results['train_score'].mean() * 100
       test_score = results['test_score'].mean() * 100
       tabela = { "MAX_DEPTH" : MAX_DEPTH, "MIN_SAMPLES_LEAF" : MIN_SAMPLES_LEAF, "train_score" : train_score, "test_score" : test_score}
       return tabela 

    def imprime_resultados(results):
       media = results['test_score'].mean() * 100
       desvio = results['test_score'].std() * 100
       print("Accuracy médio %.2f" % media)
       print("Intervalo [%.2f, %.2f]" % (media - 2 * desvio, media + 2 * desvio))


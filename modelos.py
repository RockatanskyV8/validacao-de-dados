from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

import numpy as np
import pandas as pd

import time

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

    def arvore_decisao_2d(self, CV, GROUPS, MAX_DEPTH, MIN_SAMPLES_LEAF):
       np.random.seed(self.SEED)
       modelo = DecisionTreeClassifier(max_depth=MAX_DEPTH, min_samples_leaf=MIN_SAMPLES_LEAF)
       results = cross_validate(modelo, self.x, self.y, cv = CV, groups = GROUPS, return_train_score=True)

       train_score = results['train_score'].mean() * 100
       test_score  = results['test_score'].mean() * 100

       tabela = { "MAX_DEPTH" : MAX_DEPTH, "MIN_SAMPLES_LEAF" : MIN_SAMPLES_LEAF, "train_score" : train_score, "test_score" : test_score}
       return tabela 

    def arvore_decisao_3d(self, CV, GROUPS, MAX_DEPTH, MIN_SAMPLES_LEAF, MIN_SAMPLES_SPLIT):
       np.random.seed(self.SEED)
       modelo = DecisionTreeClassifier(max_depth=MAX_DEPTH, min_samples_leaf=MIN_SAMPLES_LEAF, min_samples_split=MIN_SAMPLES_SPLIT)
       results = cross_validate(modelo, self.x, self.y, cv = CV, groups = GROUPS, return_train_score=True)

       fit_time    = results['fit_time'].mean()
       score_time  = results['score_time'].mean()
       train_score = results['train_score'].mean() * 100
       test_score  = results['test_score'].mean() * 100

       tabela = { "MAX_DEPTH" : MAX_DEPTH, "MIN_SAMPLES_LEAF" : MIN_SAMPLES_LEAF, 
                  "train_score" : train_score, "test_score" : test_score, 
                  "fit_time" : fit_time, "score_time" : score_time }
       return tabela

class Explorador:

    def __init__(self, seed, espaco_de_parametros, x, y):
        self.SEED = seed
        self.espaco_de_parametros = espaco_de_parametros
        self.x = x
        self.y = y

    def busca_grid(self, CV, GROUPS):
        np.random.seed(self.SEED)
        busca = GridSearchCV(DecisionTreeClassifier(), self.espaco_de_parametros, cv = CV)
        busca.fit(self.x, self.y, groups = GROUPS)
        return busca

    def busca_random(self, CV, N_ITER):
        busca = RandomizedSearchCV(DecisionTreeClassifier(), self.espaco_de_parametros, n_iter = N_ITER, cv = CV, random_state = SEED)
        busca.fit(self.x, self.y)
        tac = time.time()
        return busca


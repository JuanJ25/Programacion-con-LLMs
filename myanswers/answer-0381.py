import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

def optimizar_svm_grid(X, y, lista_c, cv_folds):

    modelo = SVC()

    param_grid = {'C': lista_c}

    grid = GridSearchCV(modelo, param_grid, cv=cv_folds)

    grid.fit(X, y)

    resultado = {
        'mejor_c': grid.best_params_['C'],
        'mejor_score': grid.best_score_,
        'modelo_final': grid.best_estimator_
    }

    return resultado
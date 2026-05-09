import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

def ensamble_votacion_suave(df, feature_cols, target_col):

    # Separar X e y
    X = df[feature_cols]

    y = df[target_col]

    # Crear modelos
    rf = RandomForestClassifier(random_state=42)

    svc = SVC(
        probability=True,
        random_state=42
    )

    # Crear ensamble
    ensamble = VotingClassifier(
        estimators=[('rf', rf), ('svc', svc)],
        voting='soft'
    )

    # Entrenar
    ensamble.fit(X, y)

    # Accuracy
    acc = accuracy_score(
        y,
        ensamble.predict(X)
    )

    return (ensamble, acc)
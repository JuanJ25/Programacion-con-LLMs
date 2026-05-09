import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

def analizar_sentimiento_resenas(df_resenas):

    # Copia del dataframe
    df = df_resenas.copy()

    # Limpieza
    df['review'] = df['review'].str.lower()

    df['review'] = df['review'].replace(
        r'[.,!?]',
        '',
        regex=True
    )

    # Vectorización
    vectorizer = CountVectorizer()

    X = vectorizer.fit_transform(df['review'])

    y = df['sentiment']

    # División train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42
    )

    # Modelo
    model = MultinomialNB()

    model.fit(X_train, y_train)

    # Accuracy
    accuracy = model.score(X_test, y_test)

    return (vectorizer, model, accuracy)
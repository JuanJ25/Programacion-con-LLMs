import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import random

def generar_caso_de_uso_preparar_texto_soporte():
    """
    Genera un caso de uso aleatorio para la función preparar_texto_soporte.
    """
    opciones_texto = [
        "Soporte técnico urgente", "Error en el sistema de pagos",
        "No puedo acceder a mi cuenta", "Lentitud en la plataforma",
        "Solicitud de cambio de clave", "Falla en la base de datos"
    ]
    
    n_rows = random.randint(5, 10)
    col_nombre = "mensaje_usuario"
    df = pd.DataFrame({
        col_nombre: [random.choice(opciones_texto) for _ in range(n_rows)]
    })
    
    max_palabras = random.randint(4, 8)
    
    # --- Cálculo del OUTPUT esperado (Ground Truth) ---
    # 1. Minúsculas
    text_processed = df[col_nombre].str.lower()
    # 2. Vectorizador (sin stopwords según restricción)
    vectorizer = CountVectorizer(max_features=max_palabras)
    # 3. Matriz densa
    output_matrix = vectorizer.fit_transform(text_processed).toarray()
    
    input_data = {
        'df': df,
        'col_texto': col_nombre,
        'max_palabras': max_palabras
    }
    
    return input_data, output_matrix

# --- CASO DE PRUEBA ---
if __name__ == "__main__":
    input_data, output_data = generar_caso_de_uso_preparar_texto_soporte()
    
    print("\n" + "="*50)
    print("📥 INPUT (Tickets de Soporte)")
    print("="*50)
    print(f"Límite de vocabulario (max_palabras): {input_data['max_palabras']}")
    print("\nDataFrame original:")
    print(input_data['df'])
    
    print("\n" + "="*50)
    print("📤 OUTPUT (Matriz de Frecuencias de Palabras)")
    print("="*50)
    print(output_data)
    
    print("\n" + "-"*50)
    print(f"📊 RESUMEN: {output_data.shape[0]} tickets convertidos a {output_data.shape[1]} variables numéricas.")
    print("-"*50)
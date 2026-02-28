import numpy as np
from sklearn.ensemble import IsolationForest
import random

def generar_caso_de_uso_detectar_fraude_multivariado():
    """
    Genera un caso de uso aleatorio para la función detectar_fraude_multivariado.
    """
    n_samples = random.randint(50, 100)
    n_features = random.randint(2, 4)
    
    # Generar matriz X aleatoria
    X = np.random.randn(n_samples, n_features)
    contaminacion = random.uniform(0.01, 0.2)
    
    # --- Cálculo del OUTPUT esperado (Ground Truth) ---
    model = IsolationForest(contamination=contaminacion, random_state=42)
    preds = model.fit_predict(X)
    X_clean = X[preds == 1]
    
    input_data = {
        'X': X,
        'contaminacion': contaminacion
    }
    
    return input_data, X_clean


# --- BLOQUE DE PRUEBA 0002 (VERSIÓN COMPLETA) ---
if __name__ == "__main__":
    input_data, output_data = generar_caso_de_uso_detectar_fraude_multivariado()
    
    print("\n" + "="*50)
    print("📥 INPUT COMPLETO (Diccionario de argumentos)")
    print("="*50)
    # Mostramos las claves y los valores reales que recibirá la función
    for clave, valor in input_data.items():
        print(f"\n🔑 CLAVE: {clave}")
        print(f"📄 VALOR:\n{valor}")
    
    print("\n" + "="*50)
    print("📤 OUTPUT COMPLETO (Resultado esperado)")
    print("="*50)
    # Mostramos la matriz resultante después de eliminar los outliers
    print(output_data)
    
    print("\n" + "-"*50)
    print(f"📊 RESUMEN TÉCNICO:")
    print(f"Filas originales: {input_data['X'].shape[0]}")
    print(f"Filas resultantes: {output_data.shape[0]}")
    print(f"Anomalías eliminadas: {input_data['X'].shape[0] - output_data.shape[0]}")
    print("-"*50)
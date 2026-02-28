import numpy as np
from sklearn.cluster import FeatureAgglomeration
import random

def generar_caso_de_uso_comprimir_sensores_correlacionados():
    """
    Genera un caso de uso aleatorio para la función comprimir_sensores_correlacionados.
    """
    n_samples = random.randint(10, 30)
    n_features = random.randint(6, 12)
    
    X = np.random.rand(n_samples, n_features)
    n_clusters = random.randint(2, 5)
    
    # --- Cálculo del OUTPUT esperado (Ground Truth) ---
    model = FeatureAgglomeration(n_clusters=n_clusters)
    X_transformed = model.fit_transform(X)
    
    input_data = {
        'X': X,
        'n_clusters': n_clusters
    }
    
    return input_data, X_transformed


# --- CASO DE PRUEBA ---
if __name__ == "__main__":
    input_data, output_data = generar_caso_de_uso_comprimir_sensores_correlacionados()
    
    print("\n" + "="*50)
    print("📥 INPUT (Argumentos)")
    print("="*50)
    print(f"Número de grupos deseados (n_clusters): {input_data['n_clusters']}")
    print(f"Matriz original X (Forma {input_data['X'].shape}):")
    print(input_data['X'])
    
    print("\n" + "="*50)
    print("📤 OUTPUT (Matriz Comprimida)")
    print("="*50)
    print(output_data)
    print(f"\nForma final: {output_data.shape}")
import pandas as pd
import numpy as np
import random

def generar_caso_de_uso_identificar_rutas_criticas():
    """
    Genera un caso de uso aleatorio para la función identificar_rutas_criticas.
    """
    n_rows = random.randint(20, 50)
    
    # Generar datos aleatorios de tiempos
    t_estimado = np.random.uniform(15, 60, n_rows)
    # Generar retrasos (algunos negativos para representar entregas adelantadas)
    retrasos_base = np.random.normal(loc=5, scale=10, size=n_rows)
    t_real = t_estimado + retrasos_base
    
    df = pd.DataFrame({
        'tiempo_estimado': t_estimado,
        'tiempo_real': t_real
    })
    
    percentil = random.choice([70, 75, 80, 85, 90, 95])
    
    # --- Cálculo del OUTPUT esperado (Ground Truth) ---
    df_copy = df.copy()
    df_copy['retraso'] = df_copy['tiempo_real'] - df_copy['tiempo_estimado']
    umbral = np.percentile(df_copy['retraso'], percentil)
    df_filtered = df_copy[df_copy['retraso'] > umbral]
    output_df = df_filtered.sort_values(by='retraso', ascending=False)
    
    input_data = {
        'df': df,
        'percentil': percentil
    }
    
    return input_data, output_df



    # --- CASO DE PRUEBA ---
if __name__ == "__main__":
    # 1. Llamamos a la función generadora
    input_dict, output_esperado = generar_caso_de_uso_identificar_rutas_criticas()
    
    # 2. Mostramos los resultados en la terminal
    print("📌 CASO DE USO GENERADO")
    print("-" * 30)
    print(f"Percentil a evaluar: {input_dict['percentil']}")
    print("\n--- DATOS DE ENTRADA (DF ORIGINAL) ---")
    print(input_dict['df'].head()) # Mostramos las primeras 5 filas
    
    print("\n--- SALIDA ESPERADA (FILTRADA Y ORDENADA) ---")
    if output_esperado.empty:
        print("No se encontraron rutas críticas para este percentil.")
    else:
        print(output_esperado)
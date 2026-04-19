import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

orden_edades = ['nummenor1anio', 'num1a4anios', 'num5a14anios', 'num15a64anios', 'num65omas']

def generar_informe_eda(df: pd.DataFrame):
    print("=== INFORME DE ANÁLISIS EXPLORATORIO (EDA) ===")
    print(f"Tamaño del dataset: {df.shape[0]} filas y {df.shape[1]} columnas")
    print("-" * 50)


    # [0.1] VALIDACIÓN DE INTEGRIDAD (DATA QUALITY)
    print("--- VALIDACIÓN DE INTEGRIDAD DE DATOS ---")
    duplicados = df.duplicated().sum()
    nulos_totales = df.isnull().sum().sum()
    
    resumen_calidad = pd.DataFrame({
        'Métrica': ['Filas Duplicadas', 'Valores Nulos (NaN)', 'Total de Registros'],
        'Resultado': [duplicados, nulos_totales, len(df)]
    })
    print(resumen_calidad.to_string(index=False))
    print("-" * 40)

    # [0.2] VOLUMEN PROMEDIO DE PACIENTES POR GRUPO ETARIO
    print("\n[0] VOLUMEN PROMEDIO DE PACIENTES POR GRUPO ETARIO")
    
    cols_edad = ['nummenor1anio', 'num1a4anios', 'num5a14anios', 'num15a64anios', 'num65omas']
    etiquetas_finales = ['< 1 año', '1-4 años', '5-14 años', '15-64 años', '> 65 años']
    
    promedios_edad = df[cols_edad].mean()

    plt.figure(figsize=(12, 6))
    
    # CORRECCIÓN DEL WARNING: Asignamos x a hue y legend=False
    ax = sns.barplot(
        x=etiquetas_finales, 
        y=promedios_edad.values, 
        hue=etiquetas_finales,  # Esto quita el FutureWarning
        palette="viridis",
        legend=False            # Evita que salga una leyenda que no necesitamos
    )

    # Añadir etiquetas de valor sobre las barras
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.2f}', 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha = 'center', va = 'center', 
                    xytext = (0, 9), 
                    textcoords = 'offset points',
                    fontweight='bold')

    plt.title('Promedio de Pacientes por Registro según Rango Etario', fontsize=14)
    plt.ylabel('Cantidad Promedio de Personas')
    plt.xlabel('Rango de Edad')
    sns.despine()
    
    plt.tight_layout()
    plt.show()

    # 1. Preparación de datos temporales para el gráfico superpuesto
    df_temp = df.copy()
    df_temp['fechaatenciontexto'] = pd.to_datetime(df_temp['fechaatenciontexto'])
    df_temp['Mes'] = df_temp['fechaatenciontexto'].dt.month
    df_temp['Año'] = df_temp['fechaatenciontexto'].dt.year

    # 2. Configuración de visualizaciones
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))

    # A. Distribución de Triage (Sin Desconocidos y sin Warning)
    df_triage = df[df['prioridadtriage'] != 'Desconocido']
    sns.countplot(
        data=df_triage, x='prioridadtriage', ax=axes[0, 0], 
        hue='prioridadtriage', palette='viridis', legend=False,
        order=['C1', 'C2', 'C3', 'C4', 'C5']
    )
    axes[0, 0].set_title('Atenciones por Nivel de Triage (Excluye Desconocidos)')

    # B. Top 5 Causas (Sumando NumTotal para que las barras varíen)
    top_causas = df.groupby('causa')['numtotal'].sum().sort_values(ascending=False).head(5)
    sns.barplot(
        x=top_causas.values, y=top_causas.index, ax=axes[0, 1], 
        hue=top_causas.index, palette='magma', legend=False
    )
    axes[0, 1].set_title('Top 5 Causas por Cantidad Total de Pacientes')

    # C. Boxplot de Pacientes Totales (Outliers)
    sns.boxplot(x=df['numtotal'], ax=axes[1, 0], color='#69d1a5')
    axes[1, 0].set_title('Distribución y Outliers de NumTotal')

    # D. Evolución Temporal Superpuesta (Año tras Año)
    # Agrupamos por Año y Mes sumando el total de pacientes
    df_evolucion = df_temp.groupby(['Año', 'Mes'])['numtotal'].sum().reset_index()
    
    sns.lineplot(
        data=df_evolucion, x='Mes', y='numtotal', hue='Año', 
        marker='o', ax=axes[1, 1], palette='tab10'
    )
    axes[1, 1].set_title('Comparativa Mensual de Atenciones (2023 - 2025)')
    axes[1, 1].set_xticks(range(1, 13))
    axes[1, 1].set_xticklabels(['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic'])

    plt.tight_layout()
    plt.show()

    # 3. Análisis de Correlación (Edades)
    print("\n[4] CORRELACIÓN ENTRE RANGOS ETARIOS")
    cols_edad = ['nummenor1anio', 'num1a4anios', 'num5a14anios', 'num15a64anios', 'num65omas']
    corr = df[cols_edad].corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap='RdYlGn', fmt=".2f")
    plt.title('Mapa de Calor: Correlación de Edades')
    plt.show()

# [5] RELACIÓN CAUSA VS RANGO DE EDAD: VOLUMEN VS INTENSIDAD
    print("\n[5] ANALIZANDO RELACIÓN CAUSA VS RANGO DE EDAD...")
    
    cols_edad = ['nummenor1anio', 'num1a4anios', 'num5a14anios', 'num15a64anios', 'num65omas']
    
    # Agrupación base (Suma de personas)
    causa_edad = df.groupby('causa')[cols_edad].sum()

    # 1. Normalización por FILA (Perspectiva de la Enfermedad)
    # "De 100 personas con Bronquitis, ¿qué % son bebés, adultos, etc.?"
    causa_edad_fila = causa_edad.div(causa_edad.sum(axis=1), axis=0) * 100

    # 2. Normalización por COLUMNA (Perspectiva del Grupo Etario)
    # "De 100 bebés que van a urgencias, ¿qué % va por Bronquitis?"
    causa_edad_col = causa_edad.div(causa_edad.sum(axis=0), axis=1) * 100

    # Configuración de los dos gráficos
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 10))

    # Mapa 1: Distribución por Causa (Fila)
    sns.heatmap(causa_edad_fila, annot=True, fmt=".1f", cmap="YlGnBu", ax=ax1)
    ax1.set_title('¿Quiénes se enferman de esto?\n(% de cada edad dentro de la misma causa)', fontsize=14)
    ax1.set_xlabel('Rangos de Edad')
    ax1.set_ylabel('Causa de Atención')

    # Mapa 2: Peso dentro del Grupo (Columna)
    sns.heatmap(causa_edad_col, annot=True, fmt=".1f", cmap="YlOrRd", ax=ax2)
    ax2.set_title('¿De qué se enferma este grupo?\n(% de importancia de la causa para esa edad)', fontsize=14)
    ax2.set_xlabel('Rangos de Edad')
    ax2.set_ylabel('') # Ocultamos para no repetir

    plt.tight_layout()
    plt.show()

    # [6] TENDENCIAS TEMPORALES POR GRUPO DE EDAD (Normalizado)
    print("\n[6] ANALIZANDO ESTACIONALIDAD POR GRUPO DE EDAD...")
    
    # Preparamos los datos agrupando por mes
    df_est = df.copy()
    df_est['mes'] = df_est['fechaatenciontexto'].dt.month
    cols_edad = ['nummenor1anio', 'num1a4anios', 'num5a14anios', 'num15a64anios', 'num65omas']
    
    # Sumamos pacientes por mes y edad
    estacionalidad = df_est.groupby('mes')[cols_edad].sum()

    # NORMALIZACIÓN: Dividimos cada columna por su valor máximo
    # Así todas las líneas oscilan entre 0 y 1 para comparar formas de curvas
    est_norm = estacionalidad.div(estacionalidad.max())

    plt.figure(figsize=(15, 6))
    sns.lineplot(data=est_norm, dashes=False, marker='o', linewidth=2)

    plt.title('Tendencia Estacional: ¿Cuándo ocurren los picos de atención por edad?\n(Valores normalizados para comparar curvas)')
    plt.xlabel('Mes del Año')
    plt.ylabel('Intensidad de Consultas (0 a 1)')
    plt.xticks(range(1, 13), ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic'])
    plt.grid(True, alpha=0.3)
    plt.legend(title='Grupo Etario', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

    # [7] ANÁLISIS ECONÓMICO: COSTO POR PACIENTE INDIVIDUAL POR ESTABLECIMIENTO
    print("\n[7] ANALIZANDO COSTOS UNITARIOS POR ESTABLECIMIENTO...")
    
    # 1. Creamos una columna temporal para el costo unitario (Costo Fila / Total Pacientes Fila)
    # Evitamos división por cero por seguridad
    df['costo_unitario'] = df['costoatencionclp'] / df['numtotal'].replace(0, 1)

    # 2. Ahora agrupamos por hospital, pero promediamos el COSTO UNITARIO
    costo_unitario_hospital = df.groupby('establecimientoglosa')['costo_unitario'].mean().sort_values(ascending=False)

    plt.figure(figsize=(12, 8))
    sns.barplot(
        x=costo_unitario_hospital.values, 
        y=costo_unitario_hospital.index, 
        hue=costo_unitario_hospital.index,
        palette='viridis', # Cambié a viridis para diferenciarlo del anterior
        legend=False
    )

    # 3. La línea de promedio también debe ser sobre el costo unitario
    promedio_unitario_general = df['costo_unitario'].mean()
    plt.axvline(promedio_unitario_general, color='red', linestyle='--', 
                label=f'Promedio General: ${promedio_unitario_general:,.0f}')

    plt.title('Costo Promedio POR PACIENTE según Establecimiento', fontsize=14)
    plt.xlabel('Costo por Persona Atendida ($ CLP)')
    plt.ylabel('Establecimiento')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

    # Eliminamos la columna temporal para no ensuciar el DataFrame original si lo sigues usando
    df.drop(columns=['costo_unitario'], inplace=True)

   
    # [8] ANÁLISIS ECONÓMICO: COSTO PROMEDIO POR CAUSA
    print("\n[9] ANALIZANDO COSTO PROMEDIO POR CAUSA...")
    
    # Calculamos el costo promedio por cada causa
    # Agrupamos por causa y sacamos la media del costo de atención
    costo_causa = df.groupby('causa')['costoatencionclp'].mean().sort_values(ascending=False)

    plt.figure(figsize=(12, 8))
    sns.barplot(
        x=costo_causa.values, 
        y=costo_causa.index, 
        hue=costo_causa.index,
        palette='rocket',
        legend=False
    )

    plt.title('Costo Promedio de Atención según Causa Médica')
    plt.xlabel('Costo Promedio por Atención ($ CLP)')
    plt.ylabel('Causa')

    # Añadimos etiquetas de valor para ver los montos exactos
    for i, v in enumerate(costo_causa.values):
        plt.text(v + (v * 0.01), i, f'${v:,.0f}', va='center', fontsize=10)

    plt.tight_layout()
    plt.show()

    # [9] ESTACIONALIDAD MENSUAL PROMEDIO POR CAUSA
    print("\n[12] ANALIZANDO COMPORTAMIENTO MENSUAL PROMEDIO...")

    df_est = df.copy()
    # Extraemos el mes y el nombre del mes para el eje X
    df_est['mes_num'] = df_est['fechaatenciontexto'].dt.month
    
    # Agrupamos por Mes y Causa para obtener el promedio de pacientes
    # Usamos mean() para que sea un "mes típico"
    mensual_causa = df_est.groupby(['mes_num', 'causa'])['numtotal'].mean().unstack()

    # Graficar
    plt.figure(figsize=(14, 7))
    
    # Dibujamos una línea por cada causa
    for causa in mensual_causa.columns:
        plt.plot(mensual_causa.index, mensual_causa[causa], marker='o', linewidth=2, label=causa)

    # Configuración estética
    plt.title('Promedio Mensual de Consultas: Ciclo Anual por Enfermedad', fontsize=15)
    plt.xlabel('Mes del Año')
    plt.ylabel('Promedio de Pacientes por Registro')
    
    # Forzamos que el eje X muestre los nombres de los meses
    meses_nombres = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']
    plt.xticks(range(1, 13), meses_nombres)
    
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(title='Causa Médica', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Resaltar visualmente el invierno
    plt.axvspan(6, 8, color='blue', alpha=0.05, label='Invierno')
    
    plt.tight_layout()
    plt.show()

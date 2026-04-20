import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def realizar_diagnostico_inicial(df: pd.DataFrame):
    """
    Realiza un escaneo rápido de la calidad y consistencia del dataset 
    antes de cualquier proceso de limpieza o transformación.
    """
    print("--- DIAGNÓSTICO INICIAL DE CALIDAD DE DATOS ---")
    
    # 1. Estructura Básica
    print(f"\n[1] Dimensiones: {df.shape[0]} filas y {df.shape[1]} columnas.")
    
    # 2. Análisis de Nulos, Tipos y DUPLICADOS
    print("\n[2] Análisis de Integridad Superficial:")
    
    duplicados = df.duplicated().sum()
    nulos_totales = df.isnull().sum().sum()
    
    print(f"- Filas duplicadas detectadas: {duplicados}")
    
    if nulos_totales > 0:
        print(f"- Valores nulos detectados: {nulos_totales}")
        info_calidad = pd.DataFrame({
            'Tipo': df.dtypes,
            'Nulos': df.isnull().sum(),
            '% Nulos': (df.isnull().sum() / len(df)) * 100
        })
        print(info_calidad[info_calidad['Nulos'] > 0])
    else:
        print("Integridad: No se detectan valores nulos.")

    # 3. Verificación de Integridad Relacional
    print("\n[3] Verificando Consistencia Relacional (Suma de Edades vs Total):")
    # Asegúrate de que estos nombres coincidan EXACTAMENTE con tu CSV
    cols_edad = ['NumMenor1Anio', 'Num1a4Anios', 'Num5a14Anios', 'Num15a64Anios', 'Num65oMas']
    col_total = 'NumTotal'
    
    try:
        suma_edades = df[cols_edad].sum(axis=1)
        descalces = (suma_edades != df[col_total]).sum()
        
        if descalces == 0:
            print("EXCELENTE: El 100% de los registros son coherentes (Suma edades == Total).")
        else:
            print(f"ATENCIÓN: Se detectaron {descalces} filas donde la suma no calza.")
    except KeyError as e:
        print(f"Error de nombres: No se encontró la columna {e}")

    # 4. Visualización de Dispersión Inicial
    print("\n[4] Generando Boxplot de Dispersión Original...")
    try:
        plt.figure(figsize=(12, 5))
        df[cols_edad + [col_total]].boxplot()
        plt.title("Dispersión Original de Pacientes (Detección visual de Outliers)")
        plt.xticks(rotation=45)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.show()
    except:
        print("No se pudo generar el boxplot. Verifica los nombres de las columnas.")

    # 5. Muestra Aleatoria
    print("\n[5] Muestra aleatoria de datos para inspección manual:")
    print(df.sample(min(5, len(df)))) 
    print("-" * 50)

# --- FUNCIÓN AUXILIAR PARA ZOOMS ---
def graficar_zoom_outliers(df_base, columna, titulo_grupo, color_pal, resaltado_rojo=False):
    Q1, Q3 = df_base[columna].quantile(0.25), df_base[columna].quantile(0.75)
    limite = Q3 + 1.5 * (Q3 - Q1)
    df_out = df_base[df_base[columna] > limite].copy()
    
    plt.figure(figsize=(15, 5))
    sns.stripplot(data=df_out, x='Mes', y=columna, hue='Año', dodge=True, alpha=0.6, palette=color_pal, s=8, jitter=0.25)
    plt.title(f'Picos de Atención Crítica: {titulo_grupo}\n({len(df_out)} registros detectados como Outliers > {limite:.1f} pacientes)', fontsize=13)
    plt.xticks(range(0, 12), ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic'])
    plt.ylabel('Pacientes Atendidos')
    color_vspan = 'red' if resaltado_rojo else 'gray'
    plt.axvspan(4.5, 7.5, color=color_vspan, alpha=0.07, label='Periodo Crítico Invierno')
    plt.legend(title='Año', bbox_to_anchor=(1.05, 1))
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()

# --- FUNCIÓN PRINCIPAL DEL INFORME ---
def generar_informe_eda(df: pd.DataFrame):
    print("=== INFORME DE ANÁLISIS EXPLORATORIO (EDA) ===")
    print(f"Dataset: {df.shape[0]} filas | {df.shape[1]} columnas")
    print("-" * 50)

    # [0] INTEGRIDAD Y CALIDAD
    duplicados, nulos = df.duplicated().sum(), df.isnull().sum().sum()
    print(f"--- CALIDAD DE DATOS ---\nDuplicados: {duplicados} | Nulos: {nulos}\n")

    # Preparación temporal y económica
    df_temp = df.copy()
    df_temp['fechaatenciontexto'] = pd.to_datetime(df_temp['fechaatenciontexto'])
    df_temp['Mes'] = df_temp['fechaatenciontexto'].dt.month
    df_temp['Año'] = df_temp['fechaatenciontexto'].dt.year
    df_temp['costo_unitario'] = df_temp['costoatencionclp'] / df_temp['numtotal'].replace(0, 1)

    cols_edad = ['nummenor1anio', 'num1a4anios', 'num5a14anios', 'num15a64anios', 'num65omas']
    etiquetas_edad = ['< 1 año', '1-4 años', '5-14 años', '15-64 años', '> 65 años']

    # --- [0.1] ESTADÍSTICAS DESCRIPTIVAS Y OUTLIERS ---
    print("\n--- MÉTRICAS DE TENDENCIA CENTRAL Y ATÍPICOS ---")
    cols_analisis = ['nummenor1anio', 'num1a4anios', 'num5a14anios', 'num15a64anios', 'num65omas', 'numtotal']
    
    resumen_estadistico = []

    for col in cols_analisis:
        # Tendencia Central
        promedio = df[col].mean()
        mediana = df[col].median()
        
        # Cálculo de Outliers (IQR)
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        limite_superior = Q3 + 1.5 * IQR
        
        cantidad_outliers = (df[col] > limite_superior).sum()
        porcentaje_outliers = (cantidad_outliers / len(df)) * 100
        
        resumen_estadistico.append({
            'Variable': col,
            'Promedio': f"{promedio:.2f}",
            'Mediana': f"{mediana:.2f}",
            'Cant. Outliers': cantidad_outliers,
            '% Outliers': f"{porcentaje_outliers:.2f}%"
        })

    # Mostramos la tabla consolidada
    df_resumen = pd.DataFrame(resumen_estadistico)
    print(df_resumen.to_string(index=False))
    print("-" * 60)

    # --- BLOQUE 1: RADIOGRAFÍA GENERAL ---
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    sns.barplot(x=etiquetas_edad, y=df[cols_edad].mean().values, ax=axes[0,0], hue=etiquetas_edad, palette="viridis", legend=False)
    for p in axes[0,0].patches:
        axes[0,0].annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width()/2., p.get_height()), ha='center', va='center', xytext=(0,9), textcoords='offset points', fontweight='bold')
    axes[0,0].set_title('Promedio de Pacientes por Registro')

    df_triage = df[df['prioridadtriage'] != 'Desconocido']
    sns.countplot(data=df_triage, x='prioridadtriage', ax=axes[0,1], hue='prioridadtriage', palette='viridis', order=['C1','C2','C3','C4','C5'], legend=False)
    axes[0,1].set_title('Distribución por Nivel de Triage')

    sns.heatmap(df[cols_edad].corr(), annot=True, cmap='RdYlGn', fmt=".2f", ax=axes[1,0])
    axes[1,0].set_title('Correlación de Rangos Etarios')

    top_causas = df.groupby('causa')['numtotal'].sum().sort_values(ascending=False).head(5)
    sns.barplot(x=top_causas.values, y=top_causas.index, ax=axes[1,1], hue=top_causas.index, palette='magma', legend=False)
    axes[1,1].set_title('Top 5 Causas (Volumen Total)')
    plt.tight_layout()
    plt.show()

    # --- BLOQUE 2: PICOS DE DEMANDA ---
    plt.figure(figsize=(15, 5))
    df_box = df[cols_edad + ['numtotal']].melt(var_name='Grupo', value_name='Pacientes')
    sns.boxplot(data=df_box, x='Grupo', y='Pacientes', palette='Set3', hue='Grupo', legend=False)
    plt.xticks(range(6), etiquetas_edad + ['Total'])
    plt.title('Identificación de Outliers (Boxplots)')
    plt.show()

    graficar_zoom_outliers(df_temp, 'nummenor1anio', 'Bebés (< 1 año) - ALERTA 12%', 'rocket_r', resaltado_rojo=True)
    graficar_zoom_outliers(df_temp, 'num1a4anios', 'Infantes (1 a 4 años)', 'viridis')
    graficar_zoom_outliers(df_temp, 'num65omas', 'Adultos Mayores (> 65 años)', 'magma')

    # --- [2.3] TORTAS DE OUTLIERS ---
    grupos_int = [('nummenor1anio', 'Bebés'), ('num1a4anios', 'Infantes'), ('num65omas', 'Mayores'), ('numtotal', 'Total')]
    meses_n = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    for i, (col, tit) in enumerate(grupos_int):
        Q1, Q3 = df_temp[col].quantile(0.25), df_temp[col].quantile(0.75)
        df_out = df_temp[df_temp[col] > (Q3 + 1.5*(Q3-Q1))]
        cnt = df_out['Mes'].value_counts().reindex(range(1, 13), fill_value=0)
        axes[i].pie(cnt, labels=[meses_n[m-1] if v > sum(cnt)*0.02 else '' for m, v in cnt.items()], autopct=lambda p: f'{p:.1f}%' if p > 2 else '', startangle=140, colors=sns.color_palette("hls", 12))
        axes[i].add_artist(plt.Circle((0,0), 0.70, fc='white'))
        axes[i].set_title(f'Outliers por Mes: {tit}')
    plt.tight_layout(); plt.show()

    # --- BLOQUE 3: COSTOS ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 9))
    media_gral = df_temp['costo_unitario'].mean()
    c_hosp = df_temp.groupby('establecimientoglosa')['costo_unitario'].mean().sort_values(ascending=False)
    sns.barplot(x=c_hosp.values, y=c_hosp.index, ax=ax1, hue=c_hosp.index, palette='viridis', legend=False)
    ax1.axvline(media_gral, color='red', linestyle='--'); ax1.set_title('Costo por Establecimiento')
    c_causa = df_temp.groupby('causa')['costo_unitario'].mean().sort_values(ascending=False)
    sns.barplot(x=c_causa.values, y=c_causa.index, ax=ax2, hue=c_causa.index, palette='rocket', legend=False)
    ax2.axvline(media_gral, color='red', linestyle='--'); ax2.set_title('Costo por Causa')
    plt.show()

    # --- BLOQUE 4: DIMENSIÓN TEMPORAL RECOMPLETO ---
    print("\n[4] ANÁLISIS DE ESTACIONALIDAD...")
    
    # A. Evolución Histórica
    plt.figure(figsize=(15, 5))
    df_ev = df_temp.groupby(['Año', 'Mes'])['numtotal'].sum().reset_index()
    sns.lineplot(data=df_ev, x='Mes', y='numtotal', hue='Año', marker='o')
    plt.title('Comparativa Mensual de Atenciones (Evolución Anual)')
    plt.xticks(range(1, 13), meses_n); plt.show()

    # B. Picos Normalizados por Edad
    plt.figure(figsize=(15, 5))
    # Normalizamos (0 a 1) para que los 500 bebés no "tapen" a los otros grupos
    est_norm = df_temp.groupby('Mes')[cols_edad].sum()
    est_norm = (est_norm - est_norm.min()) / (est_norm.max() - est_norm.min())
    sns.lineplot(data=est_norm, dashes=False, marker='o', linewidth=2.5)
    plt.title('Sincronía de Picos por Edad (Escala Normalizada 0-1)')
    plt.xticks(range(1, 13), meses_n); plt.ylabel('Intensidad de Demanda'); plt.show()

    # C. Ciclo por Enfermedad
    plt.figure(figsize=(15, 6))
    m_causa = df_temp.groupby(['Mes', 'causa'])['numtotal'].mean().unstack()
    
    for c in m_causa.columns: 
        plt.plot(m_causa.index, m_causa[c], marker='o', label=c)
    
    plt.axvspan(6, 8, color='blue', alpha=0.05, label='Invierno')
    
    # --- Título y Etiquetas de Ejes ---
    plt.title('Evolución Mensual de la Demanda Asistencial por Patología y Periodo Estacional', fontsize=14)
    plt.xlabel('Mes del Año', fontsize=12)
    plt.ylabel('Promedio de Pacientes (Atenciones)', fontsize=12)
    
    plt.xticks(range(1, 13), meses_n)
    plt.legend(title='Patologías', bbox_to_anchor=(1.05, 1))
    plt.tight_layout()
    plt.show()

    # --- BLOQUE 5: ANÁLISIS POR SEXO (DIMENSIÓN DEMOGRÁFICA) ---
    print("\n[5] ANÁLISIS POR SEXO Y TENDENCIAS...")
    
    # 1. Definimos la paleta manual
    paleta_genero = {"M": "royalblue", "F": "pink"}
    
    # Filtramos registros válidos
    df_sexo = df_temp[df_temp['sexopaciente'].isin(['F', 'M'])]
    
    fig, axes = plt.subplots(1, 3, figsize=(22, 7))

    # A. Volumen Total por Fecha y Sexo
    # Usamos hue_order para asegurar que el orden de la leyenda sea consistente
    sns.lineplot(data=df_sexo.groupby(['Mes', 'sexopaciente'])['numtotal'].sum().reset_index(), 
                 x='Mes', y='numtotal', hue='sexopaciente', hue_order=['M', 'F'],
                 marker='o', ax=axes[0], palette=paleta_genero)
    axes[0].set_title('Evolución Mensual: Hombre (Azul) vs Mujer (Naranja)')
    axes[0].set_xticks(range(1, 13))
    axes[0].set_xticklabels(meses_n)

    # B. Top Causas segmentadas por Sexo
    top_5_causas = df_temp.groupby('causa')['numtotal'].sum().sort_values(ascending=False).head(5).index
    df_causa_sexo = df_sexo[df_sexo['causa'].isin(top_5_causas)]
    sns.barplot(data=df_causa_sexo, x='numtotal', y='causa', hue='sexopaciente', 
                hue_order=['M', 'F'], estimator=sum, errorbar=None, ax=axes[1], palette=paleta_genero)
    axes[1].set_title('Carga de Enfermedad por Sexo (Top 5)')

    # C. Distribución de Costos por Sexo
    sns.boxplot(data=df_sexo, x='sexopaciente', y='costo_unitario', hue='sexopaciente',
                order=['M', 'F'], palette=paleta_genero, ax=axes[2], showfliers=False)
    axes[2].axhline(media_gral, color='red', linestyle='--', label='Media Gral')
    axes[2].set_title('Distribución de Costo Unitario por Sexo')
    
    plt.tight_layout()
    plt.show()

    # --- BLOQUE 6: ANÁLISIS CRUZADO DE PATOLOGÍAS Y EDAD ---
    print("\n[6] GENERANDO PERFILES EPIDEMIOLÓGICOS...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(26, 10))

    # --- A. PERFIL ETARIO POR PATOLOGÍA (Normalizado por Fila) ---
    # Responde: "De los que tienen Bronquitis, ¿qué % son bebés?"
    df_causa_edad = df_temp.groupby('causa')[cols_edad].sum()
    df_causa_pct = df_causa_edad.div(df_causa_edad.sum(axis=1), axis=0) * 100
    
    df_causa_pct.plot(kind='barh', stacked=True, ax=ax1, colormap='viridis')
    ax1.set_title('A. Distribución de Edades por cada Patología', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Porcentaje del Grupo Etario dentro de la Enfermedad (%)')
    ax1.set_ylabel('Patología')
    ax1.legend(etiquetas_edad, title='Rango de Edad', loc='lower right', fontsize=9)

    # Añadir etiquetas al gráfico A
    for p in ax1.patches:
        width = p.get_width()
        if width > 6:
            ax1.text(p.get_x() + width/2, p.get_y() + p.get_height()/2, f'{width:.1f}%', 
                     va='center', ha='center', color='white', fontweight='bold', fontsize=8)

    # --- B. COMPOSICIÓN DE ENFERMEDADES POR EDAD (Normalizado por Columna) ---
    # Responde: "De los bebés que llegan, ¿qué % tiene bronquitis?"
    df_edad_relativo = df_causa_edad.div(df_causa_edad.sum(axis=0), axis=1) * 100
    
    df_edad_relativo.T.plot(kind='barh', stacked=True, ax=ax2, colormap='tab20')
    ax2.set_title('B. Mix de Patologías dentro de cada Grupo Etario', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Porcentaje de la Patología dentro del Rango de Edad (%)')
    ax2.set_ylabel('Grupo Etario')
    ax2.legend(title='Patologías', bbox_to_anchor=(1.05, 1))

    # Añadir etiquetas al gráfico B
    for p in ax2.patches:
        width = p.get_width()
        if width > 8:
            ax2.text(p.get_x() + width/2, p.get_y() + p.get_height()/2, f'{width:.1f}%', 
                     va='center', ha='center', color='white', fontweight='bold', fontsize=8)

    plt.tight_layout()
    plt.show()

    # --- BLOQUE 7: ANÁLISIS ESTRATÉGICO DE TRIAGE (CORREGIDO) ---
    print("\n[7] ANALIZANDO RELACIÓN ENFERMEDAD-GRAVEDAD CON SEMÁFORO DE SALUD...")
    
    df_triage_clean = df_temp[df_temp['prioridadtriage'].isin(['C1', 'C2', 'C3', 'C4', 'C5'])]
    orden_triage = ['C1', 'C2', 'C3', 'C4', 'C5']
    
    # Definimos paleta manual: C1=Rojo, C2=Naranja, C3=Amarillo, C4=Verde Claro, C5=Verde Fuerte
    colores_triage = ['#d73027', '#f46d43', '#fee08b', '#d9ef8b', '#1a9850']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 10))

    # A. TRIAGE POR ENFERMEDAD (Composición de Gravedad)
    df_causa_triage = df_triage_clean.groupby(['causa', 'prioridadtriage'])['numtotal'].sum().unstack().fillna(0)
    df_causa_triage_pct = df_causa_triage.div(df_causa_triage.sum(axis=1), axis=0) * 100

    # Graficamos con los colores correctos
    df_causa_triage_pct[orden_triage].plot(kind='barh', stacked=True, ax=ax1, color=colores_triage)
    ax1.set_title('A. Composición de Gravedad (Triage) por Patología', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Porcentaje del Nivel de Triage (%)')
    ax1.legend(title='Nivel Triage (Rojo = Grave)', bbox_to_anchor=(1.05, 1))

    # B. EVOLUCIÓN MENSUAL DEL TRIAGE
    df_mes_triage = df_triage_clean.groupby(['Mes', 'prioridadtriage'])['numtotal'].sum().unstack().fillna(0)
    df_mes_triage_pct = df_mes_triage.div(df_mes_triage.sum(axis=1), axis=0) * 100

    # Usamos los mismos colores para las líneas
    for i, nivel in enumerate(orden_triage):
        ax2.plot(df_mes_triage_pct.index, df_mes_triage_pct[nivel], marker='o', 
                 label=nivel, color=colores_triage[i], linewidth=3)

    ax2.axvspan(6, 8, color='blue', alpha=0.05, label='Invierno')
    ax2.set_title('B. Evolución de la Gravedad Relativa por Mes', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Composición Porcentual (%)')
    ax2.set_xticks(range(1, 13))
    ax2.set_xticklabels(meses_n)
    ax2.grid(True, alpha=0.3)
    ax2.legend(title='Prioridad', loc='upper left')

    plt.tight_layout()
    plt.show()

    print("-" * 50)
    print("EDA Finalizado. El dataset está listo para el informe técnico y el modelamiento.")
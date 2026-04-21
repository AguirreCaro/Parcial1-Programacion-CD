import pandas as pd

# Constante para normalizar el Sexo del Paciente
SEXO_VALIDO = {
    "femenino": "F",
    "f": "F",
    "masculino": "M",
    "m": "M"
}

# Constante para normalizar los niveles de Triage
TRIAGE_NORMALIZADO = {
    "c1": "C1",
    "c2": "C2",
    "c3": "C3",
    "c4": "C4",
    "c5": "C5"
}

# Constantes para límites numéricos
PACIENTES_MIN = 0
COSTO_MIN = 0

# Constante para columnas requeridas (dejamos los nombres originales para la validación inicial)
COLUMNAS_REQUERIDAS = {'EstablecimientoCodigo', 'EstablecimientoGlosa', 'RegionCodigo',
       'RegionGlosa', 'ComunaCodigo', 'ComunaGlosa', 'ServicioSaludCodigo',
       'ServicioSaludGlosa', 'TipoEstablecimiento','DependenciaAdministrativa',
       'NivelAtencion', 'TipoUrgencia', 'Latitud','Longitud', 'NivelComplejidad',
       'Anio', 'SemanaEstadistica','OrdenCausa', 'Causa', 'NumTotal', 'NumMenor1Anio', 'Num1a4Anios',
       'Num5a14Anios', 'Num15a64Anios', 'Num65oMas', 'FechaAtencionTexto',
       'SexoPaciente', 'PrioridadTriage', 'CostoAtencionCLP'}

# Diccionario para unificar las causas con errores de digitación
CAUSAS_CORRECCION = {
    'Bronquitis/Bronquiolitis agudaa': 'bronquitis/bronquiolitis aguda',
    'bronquitis/bronquiolitis agudaa': 'bronquitis/bronquiolitis aguda',
    'insuficiencia respirat0ria agudaa': 'insuficiencia respiratoria aguda',
    'infeccion respirat0ria agudaa alta': 'infeccion respiratoria aguda alta'
}

COLUMNAS_EDAD = [
    'nummenor1anio', 
    'num1a4anios', 
    'num5a14anios', 
    'num15a64anios', 
    'num65omas'
]

def limpiar_datos(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 1. Validar columnas requeridas (ANTES de pasarlas a minúsculas)
    columnas_faltantes = COLUMNAS_REQUERIDAS - set(df.columns)
    if columnas_faltantes:
        raise ValueError(f"Faltan columnas requeridas: {', '.join(columnas_faltantes)}")

    # 2. Normalizar nombres de columnas a minúsculas
    df.columns = df.columns.str.lower().str.strip()

    # 3. Limpiar espacios en columnas de texto y manejar nulos de texto
    df['causa'] = df['causa'].astype(str).str.lower().str.strip()
    df['causa'] = df['causa'].replace(CAUSAS_CORRECCION)

    columnas_texto = ['establecimientoglosa', 'regionglosa', 'comunaglosa', 'serviciosaludglosa',
                      'tipoestablecimiento', 'dependenciaadministrativa', 'nivelatencion',
                      'tipourgencia', 'nivelcomplejidad']

    for col in columnas_texto:
        df[col] = df[col].astype(str).str.lower().str.strip().fillna("sin información")

    # 4. Normalizar Sexo Paciente
    df["sexopaciente"] = (
        df["sexopaciente"]
        .str.lower()
        .str.strip()
        .map(SEXO_VALIDO)
    )
    df["sexopaciente"] = df["sexopaciente"].fillna("Desconocido") # Rellenar nulos de sexo

    # 5. Normalizar prioridad triage quitando guiones, espacios y mapeando
    df["prioridadtriage"] = (
        df["prioridadtriage"]
        .str.replace("-", "", regex=False)
        .str.replace(" ", "", regex=False)
        .str.lower()
        .str.strip()
        .map(TRIAGE_NORMALIZADO)
    )
    df["prioridadtriage"] = df["prioridadtriage"].fillna("Desconocido") # Los 'c-3', 'c3' se mapean, el resto queda nulo/desconocido

    # 6. Validar y normalizar valores numéricos (PASO CRÍTICO)
    cols_numericas = ['numtotal', 'costoatencionclp'] + COLUMNAS_EDAD
    for col in cols_numericas:
        # Forzamos conversión a número antes de cualquier suma
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # Calculamos la suma después de asegurar que todo es numérico
    suma_calculada = df[COLUMNAS_EDAD].sum(axis=1)
    
    # Aplicamos el filtro: Solo mantenemos las filas donde coinciden
    df = df[df['numtotal'] == suma_calculada].copy()

    # Filtros de mínimos (negativos)
    df = df[df['numtotal'] >= PACIENTES_MIN]
    df = df[df['costoatencionclp'] >= COSTO_MIN]

    # 7. Manejar Coordenadas (Latitud y Longitud vacíos)
    df['latitud'] = pd.to_numeric(df['latitud'], errors='coerce').fillna(0)
    df['longitud'] = pd.to_numeric(df['longitud'], errors='coerce').fillna(0)

    # 8. Convertir fechas y eliminar registros sin fecha
    df["fechaatenciontexto"] = pd.to_datetime(df["fechaatenciontexto"], errors="coerce", dayfirst=True, format="mixed")
    
    # Eliminamos las filas donde la fecha no se pudo convertir (NaT)
    df = df.dropna(subset=["fechaatenciontexto"])

    # 9. Eliminar duplicados
    df = df.drop_duplicates()

    columnas_finales = [
        'establecimientocodigo', 'establecimientoglosa', 'anio', 'semanaestadistica',
        'causa', 'numtotal', 'nummenor1anio', 'num1a4anios', 
        'num5a14anios', 'num15a64anios', 'num65omas', 
        'fechaatenciontexto', 'sexopaciente', 'prioridadtriage', 'costoatencionclp'
    ]
    
    # Solo devolvemos las columnas que nos interesan
    df = df[columnas_finales].copy()

    return df
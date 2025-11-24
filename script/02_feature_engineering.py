"""
==================================================
PROYECTO CHURN - COTEL
Script 2: Feature Engineering
==================================================
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("FEATURE ENGINEERING - COTEL")
print("="*60)

# 1. CARGAR DATOS LIMPIOS
print("\n1. Cargando datos...")
clientes = pd.read_csv('../output/clientes_limpio.csv')
trafico_jul = pd.read_csv('../data/trafico_julio_corregido.csv')
trafico_ago = pd.read_csv('../data/trafico_agosto_corregido.csv')
trafico_sep = pd.read_csv('../data/trafico_sept_corregido.csv')

print(f"✓ Clientes: {len(clientes):,}")
print(f"✓ Llamadas Julio: {len(trafico_jul):,}")
print(f"✓ Llamadas Agosto: {len(trafico_ago):,}")
print(f"✓ Llamadas Septiembre: {len(trafico_sep):,}")

# 2. AGREGAR TRÁFICO POR TELÉFONO Y MES
print("\n" + "="*60)
print("2. AGREGANDO TRÁFICO POR TELÉFONO")
print("="*60)

def agregar_trafico(df, mes_nombre):
    """Agregar tráfico por teléfono"""
    
    print(f"   Procesando {len(df):,} registros...")
    
    # Crear copia para no modificar original
    df_temp = df.copy()
    
    # Convertir fechahora manejando formatos mixtos
    df_temp['fechahora'] = pd.to_datetime(df_temp['fechahora'], format='mixed', errors='coerce')
    
    # Eliminar filas con fechas inválidas
    registros_antes = len(df_temp)
    df_temp = df_temp[df_temp['fechahora'].notna()]
    registros_despues = len(df_temp)
    
    if registros_antes != registros_despues:
        print(f"   ⚠️  Eliminados {registros_antes - registros_despues:,} registros con fechas inválidas")
    
    # Agregar por teléfono
    agg = df_temp.groupby('telefono').agg({
        'duracion': ['count', 'sum', 'mean', 'std', 'max'],
        'destino': 'nunique'
    }).reset_index()
    
    # Aplanar columnas
    agg.columns = ['telefono', 
                   f'llamadas_{mes_nombre}',
                   f'minutos_total_{mes_nombre}',
                   f'duracion_promedio_{mes_nombre}',
                   f'duracion_std_{mes_nombre}',
                   f'duracion_max_{mes_nombre}',
                   f'destinos_unicos_{mes_nombre}']
    
    # Convertir segundos a minutos
    agg[f'minutos_total_{mes_nombre}'] = (agg[f'minutos_total_{mes_nombre}'] / 60).round(2)
    agg[f'duracion_promedio_{mes_nombre}'] = (agg[f'duracion_promedio_{mes_nombre}'] / 60).round(2)
    agg[f'duracion_std_{mes_nombre}'] = (agg[f'duracion_std_{mes_nombre}'] / 60).round(2)
    agg[f'duracion_max_{mes_nombre}'] = (agg[f'duracion_max_{mes_nombre}'] / 60).round(2)
    
    # Rellenar NaN en desviación estándar (cuando solo hay 1 llamada)
    agg[f'duracion_std_{mes_nombre}'] = agg[f'duracion_std_{mes_nombre}'].fillna(0)
    
    # Calcular días activos
    df_temp['fecha'] = df_temp['fechahora'].dt.date
    dias_activos = df_temp.groupby('telefono')['fecha'].nunique().reset_index()
    dias_activos.columns = ['telefono', f'dias_activos_{mes_nombre}']
    
    agg = agg.merge(dias_activos, on='telefono', how='left')
    
    return agg

print("\n--- Procesando Julio...")
trafico_jul_agg = agregar_trafico(trafico_jul, 'jul')
print(f"✓ Teléfonos únicos: {len(trafico_jul_agg):,}")

print("\n--- Procesando Agosto...")
trafico_ago_agg = agregar_trafico(trafico_ago, 'ago')
print(f"✓ Teléfonos únicos: {len(trafico_ago_agg):,}")

print("\n--- Procesando Septiembre...")
trafico_sep_agg = agregar_trafico(trafico_sep, 'sep')
print(f"✓ Teléfonos únicos: {len(trafico_sep_agg):,}")

# 3. UNIR DATOS
print("\n" + "="*60)
print("3. UNIENDO DATASETS")
print("="*60)

# Unir clientes con tráfico
df = clientes.copy()

print(f"\nClientes iniciales: {len(df):,}")

df = df.merge(trafico_jul_agg, on='telefono', how='left')
print(f"✓ Después de unir Julio: {len(df):,} ({df['llamadas_jul'].notna().sum():,} con datos)")

df = df.merge(trafico_ago_agg, on='telefono', how='left')
print(f"✓ Después de unir Agosto: {len(df):,} ({df['llamadas_ago'].notna().sum():,} con datos)")

df = df.merge(trafico_sep_agg, on='telefono', how='left')
print(f"✓ Después de unir Septiembre: {len(df):,} ({df['llamadas_sep'].notna().sum():,} con datos)")

# 4. RELLENAR TRÁFICO CON 0 (clientes sin actividad)
print("\n" + "="*60)
print("4. RELLENANDO VALORES FALTANTES EN TRÁFICO")
print("="*60)

columnas_trafico = [col for col in df.columns if any(x in col for x in ['llamadas', 'minutos', 'duracion', 'destinos', 'dias'])]

print(f"\n✓ Rellenando {len(columnas_trafico)} columnas de tráfico con 0")
for col in columnas_trafico:
    df[col] = df[col].fillna(0)

# 5. CREAR FEATURES AGREGADAS (3 meses)
print("\n" + "="*60)
print("5. CREANDO FEATURES AGREGADAS")
print("="*60)

# Total 3 meses
df['llamadas_total_3m'] = df['llamadas_jul'] + df['llamadas_ago'] + df['llamadas_sep']
df['minutos_total_3m'] = df['minutos_total_jul'] + df['minutos_total_ago'] + df['minutos_total_sep']
df['destinos_unicos_total_3m'] = df['destinos_unicos_jul'] + df['destinos_unicos_ago'] + df['destinos_unicos_sep']
df['dias_activos_total_3m'] = df['dias_activos_jul'] + df['dias_activos_ago'] + df['dias_activos_sep']

# Promedios
df['llamadas_promedio_mes'] = (df['llamadas_total_3m'] / 3).round(2)
df['minutos_promedio_mes'] = (df['minutos_total_3m'] / 3).round(2)

# Minutos por llamada (evitar división por 0)
df['minutos_por_llamada'] = np.where(
    df['llamadas_total_3m'] > 0,
    (df['minutos_total_3m'] / df['llamadas_total_3m']).round(2),
    0
)

# Llamadas por día activo
df['llamadas_por_dia'] = np.where(
    df['dias_activos_total_3m'] > 0,
    (df['llamadas_total_3m'] / df['dias_activos_total_3m']).round(2),
    0
)

print(f"\n✓ Features agregadas creadas")
print(f"  - llamadas_total_3m: {df['llamadas_total_3m'].sum():,.0f}")
print(f"  - minutos_total_3m: {df['minutos_total_3m'].sum():,.2f}")
print(f"  - minutos_por_llamada promedio: {df[df['llamadas_total_3m']>0]['minutos_por_llamada'].mean():.2f}")

# 6. CREAR FEATURES DE TENDENCIA
print("\n" + "="*60)
print("6. CREANDO FEATURES DE TENDENCIA")
print("="*60)

# Tendencia de llamadas (Septiembre vs Julio)
df['tendencia_llamadas'] = df['llamadas_sep'] - df['llamadas_jul']

# Cambio porcentual en llamadas
df['cambio_pct_llamadas'] = np.where(
    df['llamadas_jul'] > 0,
    ((df['llamadas_sep'] - df['llamadas_jul']) / df['llamadas_jul'] * 100).round(2),
    0
)

# Tendencia de minutos
df['tendencia_minutos'] = df['minutos_total_sep'] - df['minutos_total_jul']

# Cambio porcentual en minutos
df['cambio_pct_minutos'] = np.where(
    df['minutos_total_jul'] > 0,
    ((df['minutos_total_sep'] - df['minutos_total_jul']) / df['minutos_total_jul'] * 100).round(2),
    0
)

# Variabilidad de uso (desviación estándar de llamadas entre meses)
df['variabilidad_uso'] = df[['llamadas_jul', 'llamadas_ago', 'llamadas_sep']].std(axis=1).round(2)

# Flag: Cliente con reducción de uso
df['reduccion_uso'] = ((df['tendencia_llamadas'] < -10) | (df['cambio_pct_llamadas'] < -20)).astype(int)

# Flag: Cliente sin actividad
df['sin_actividad'] = (df['llamadas_total_3m'] == 0).astype(int)

print(f"\n✓ Features de tendencia creadas")
print(f"  - Clientes con reducción de uso: {df['reduccion_uso'].sum():,} ({df['reduccion_uso'].mean()*100:.2f}%)")
print(f"  - Clientes sin actividad: {df['sin_actividad'].sum():,} ({df['sin_actividad'].mean()*100:.2f}%)")

# 7. FEATURES DEMOGRÁFICAS
print("\n" + "="*60)
print("7. PROCESANDO FEATURES DEMOGRÁFICAS")
print("="*60)

# Edad en categorías
df['edad_grupo'] = pd.cut(df['edad'], 
                          bins=[0, 30, 40, 50, 60, 70, 120], 
                          labels=['<30', '30-40', '40-50', '50-60', '60-70', '70+'],
                          include_lowest=True)

# Antigüedad en categorías (años)
df['antiguedad_anos'] = (df['antiguedad_meses'] / 12).round(1)
df['antiguedad_grupo'] = pd.cut(df['antiguedad_anos'], 
                                bins=[0, 1, 3, 5, 10, 20, 100], 
                                labels=['<1año', '1-3años', '3-5años', '5-10años', '10-20años', '20+años'],
                                include_lowest=True)

# Rellenar valores categóricos faltantes
df['sexo'] = df['sexo'].fillna('Desconocido')
df['estado_civil'] = df['estado_civil'].fillna('Desconocido')
df['ciudad'] = df['ciudad'].fillna('Desconocido')
df['zona'] = df['zona'].fillna('Desconocido')
df['edad_grupo'] = df['edad_grupo'].astype(str).replace('nan', 'Desconocido')
df['antiguedad_grupo'] = df['antiguedad_grupo'].astype(str).replace('nan', 'Desconocido')

print(f"\n✓ Features demográficas procesadas")

# 8. RESUMEN DEL DATASET FINAL
print("\n" + "="*60)
print("RESUMEN DEL DATASET FINAL")
print("="*60)

print(f"\n📊 DIMENSIONES")
print(f"  - Total registros: {len(df):,}")
print(f"  - Total features: {len(df.columns)}")

print(f"\n📞 DISTRIBUCIÓN DE CHURN")
print(df['churn'].value_counts())
print(f"  Tasa de churn: {df['churn'].mean()*100:.2f}%")

print(f"\n📈 ESTADÍSTICAS DE TRÁFICO")
print(f"  - Clientes con actividad: {(df['llamadas_total_3m'] > 0).sum():,} ({(df['llamadas_total_3m'] > 0).mean()*100:.2f}%)")
print(f"  - Clientes sin actividad: {df['sin_actividad'].sum():,} ({df['sin_actividad'].mean()*100:.2f}%)")
print(f"  - Promedio llamadas/cliente: {df['llamadas_total_3m'].mean():.2f}")
print(f"  - Promedio minutos/cliente: {df['minutos_total_3m'].mean():.2f}")

# 9. GUARDAR DATASET FINAL
print("\n" + "="*60)
print("GUARDANDO DATASET FINAL")
print("="*60)

df.to_csv('../output/dataset_final.csv', index=False)
print(f"✓ Guardado: output/dataset_final.csv")
print(f"  Tamaño: {len(df):,} registros x {len(df.columns)} columnas")

# Guardar también las primeras filas como muestra
df.head(100).to_csv('../output/dataset_muestra.csv', index=False)
print(f"✓ Guardado: output/dataset_muestra.csv (primeras 100 filas)")

# 10. ANÁLISIS RÁPIDO DE CHURN VS NO CHURN
print("\n" + "="*60)
print("ANÁLISIS: CHURN VS NO CHURN")
print("="*60)

churn_stats = df.groupby('churn')[['llamadas_total_3m', 'minutos_total_3m', 
                                     'antiguedad_meses', 'edad']].mean()
print("\n--- Promedios por grupo ---")
print(churn_stats)

print("\n" + "="*60)
print("FEATURE ENGINEERING COMPLETADO ✓")
print("="*60)
print("\n🚀 Siguiente paso: Ejecutar 03_modeling.py")
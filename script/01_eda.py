"""
==================================================
PROYECTO CHURN - COTEL
Script 1: Análisis Exploratorio de Datos (EDA)
==================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configuración
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Crear carpeta output si no existe
os.makedirs('../output/graficos', exist_ok=True)

print("="*60)
print("CARGA DE DATOS")
print("="*60)

# 1. CARGAR DATOS (ARCHIVOS CORREGIDOS)
print("\n1. Cargando CSVs corregidos...")
clientes = pd.read_csv('../data/clientes_corregido.csv', low_memory=False)
trafico_jul = pd.read_csv('../data/trafico_julio_corregido.csv')
trafico_ago = pd.read_csv('../data/trafico_agosto_corregido.csv')
trafico_sep = pd.read_csv('../data/trafico_sept_corregido.csv')

print(f"✓ Clientes: {len(clientes):,} registros")
print(f"✓ Tráfico Julio 2025: {len(trafico_jul):,} llamadas")
print(f"✓ Tráfico Agosto 2025: {len(trafico_ago):,} llamadas")
print(f"✓ Tráfico Septiembre 2025: {len(trafico_sep):,} llamadas")

# 2. VISTA PREVIA
print("\n" + "="*60)
print("ESTRUCTURA DE DATOS")
print("="*60)

print("\n--- CLIENTES (primeras 5 filas) ---")
print(clientes.head(5))

print("\n--- Columnas en clientes ---")
print(clientes.columns.tolist())

print("\n--- Tipos de datos ---")
print(clientes.dtypes)

print("\n--- Valores nulos ---")
nulos = clientes.isnull().sum()
print(nulos[nulos > 0])

print("\n--- TRÁFICO JULIO (primeras 3 filas) ---")
print(trafico_jul.head(3))

# 3. ESTADÍSTICAS DESCRIPTIVAS
print("\n" + "="*60)
print("ESTADÍSTICAS DESCRIPTIVAS - CLIENTES")
print("="*60)

print("\n--- Variables numéricas ---")
print(clientes.describe())

# 4. ANÁLISIS DE CHURN
print("\n" + "="*60)
print("ANÁLISIS DE CHURN")
print("="*60)

# Crear variable churn
clientes['churn'] = clientes['fecha_retiro'].notna().astype(int)

print(f"\nTotal clientes: {len(clientes):,}")
print(f"Clientes con churn (dieron de baja): {clientes['churn'].sum():,}")
print(f"Clientes activos: {(clientes['churn'] == 0).sum():,}")
print(f"Tasa de churn: {clientes['churn'].mean()*100:.2f}%")

# 5. GRÁFICO: Distribución de Churn
plt.figure(figsize=(10, 6))
churn_counts = clientes['churn'].value_counts()
colors = ['#2ecc71', '#e74c3c']  # Verde para activos, rojo para churn
plt.bar(['Activos (0)', 'Churn (1)'], churn_counts.values, color=colors, alpha=0.8, edgecolor='black')
plt.title('Distribución de Churn - COTEL', fontsize=16, fontweight='bold')
plt.ylabel('Cantidad de Clientes', fontsize=12)
plt.xlabel('Estado', fontsize=12)

for i, v in enumerate(churn_counts.values):
    plt.text(i, v + max(churn_counts.values)*0.02, 
             f'{v:,}\n({v/len(clientes)*100:.2f}%)', 
             ha='center', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('../output/graficos/01_distribucion_churn.png', dpi=300, bbox_inches='tight')
print("\n✓ Gráfico guardado: output/graficos/01_distribucion_churn.png")
plt.close()

# 6. ANÁLISIS DEMOGRÁFICO
print("\n" + "="*60)
print("ANÁLISIS DEMOGRÁFICO")
print("="*60)

# Convertir fechas con manejo robusto de errores
today = pd.Timestamp.now()
fecha_min_nacimiento = pd.Timestamp('1900-01-01')
fecha_min_instalacion = pd.Timestamp('1950-01-01')

# FECHA DE NACIMIENTO Y EDAD
print("\n--- Calculando EDAD ---")
clientes['fecha_nacimiento'] = pd.to_datetime(clientes['fecha_nacimiento'], errors='coerce')

# Filtrar fechas inválidas
clientes.loc[clientes['fecha_nacimiento'] < fecha_min_nacimiento, 'fecha_nacimiento'] = pd.NaT
clientes.loc[clientes['fecha_nacimiento'] > today, 'fecha_nacimiento'] = pd.NaT

# Calcular edad
clientes['edad'] = None
mask_validas = clientes['fecha_nacimiento'].notna()

if mask_validas.sum() > 0:
    clientes.loc[mask_validas, 'edad'] = (
        (today - clientes.loc[mask_validas, 'fecha_nacimiento']).dt.days / 365.25
    ).round(0)
    
    # Filtrar edades imposibles (0-120 años)
    clientes.loc[(clientes['edad'] < 0) | (clientes['edad'] > 120), 'edad'] = None

print(f"✓ Clientes con edad conocida: {clientes['edad'].notna().sum():,} ({clientes['edad'].notna().sum()/len(clientes)*100:.1f}%)")
if clientes['edad'].notna().sum() > 0:
    print("\n--- Distribución de Edad ---")
    print(clientes['edad'].describe())

# FECHA DE INSTALACIÓN Y ANTIGÜEDAD
print("\n--- Calculando ANTIGÜEDAD ---")
clientes['fecha_instalacion'] = pd.to_datetime(clientes['fecha_instalacion'], errors='coerce')

# Filtrar fechas inválidas
clientes.loc[clientes['fecha_instalacion'] < fecha_min_instalacion, 'fecha_instalacion'] = pd.NaT
clientes.loc[clientes['fecha_instalacion'] > today, 'fecha_instalacion'] = pd.NaT

# Calcular antigüedad en meses
clientes['antiguedad_meses'] = None
mask_validas = clientes['fecha_instalacion'].notna()

if mask_validas.sum() > 0:
    clientes.loc[mask_validas, 'antiguedad_meses'] = (
        (today - clientes.loc[mask_validas, 'fecha_instalacion']).dt.days / 30.44
    ).round(0)
    
    # Filtrar antigüedades imposibles
    clientes.loc[clientes['antiguedad_meses'] < 0, 'antiguedad_meses'] = None

print(f"✓ Clientes con antigüedad conocida: {clientes['antiguedad_meses'].notna().sum():,} ({clientes['antiguedad_meses'].notna().sum()/len(clientes)*100:.1f}%)")
if clientes['antiguedad_meses'].notna().sum() > 0:
    print("\n--- Distribución de Antigüedad (meses) ---")
    print(clientes['antiguedad_meses'].describe())
    print(f"Promedio: {clientes['antiguedad_meses'].mean():.1f} meses ({clientes['antiguedad_meses'].mean()/12:.1f} años)")

# DISTRIBUCIÓN POR CIUDAD
print("\n--- Distribución por Ciudad (top 10) ---")
if clientes['ciudad'].notna().sum() > 0:
    print(clientes['ciudad'].value_counts().head(10))
else:
    print("Sin datos de ciudad")

# DISTRIBUCIÓN POR TIPO PERSONERÍA
print("\n--- Distribución por Tipo Personería ---")
print(clientes['tipo_personeria'].value_counts())

# DISTRIBUCIÓN POR SEXO
print("\n--- Distribución por Sexo ---")
if clientes['sexo'].notna().sum() > 0:
    print(clientes['sexo'].value_counts())
else:
    print("Sin datos de sexo")

# 7. GRÁFICOS: Distribución de Edad y Antigüedad
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Edad - Histograma
if clientes['edad'].notna().sum() > 0:
    axes[0, 0].hist(clientes['edad'].dropna(), bins=30, edgecolor='black', alpha=0.7, color='steelblue')
    axes[0, 0].set_title('Distribución de Edad', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Edad (años)')
    axes[0, 0].set_ylabel('Frecuencia')
    axes[0, 0].axvline(clientes['edad'].mean(), color='red', linestyle='--', label=f'Media: {clientes["edad"].mean():.1f}')
    axes[0, 0].legend()
else:
    axes[0, 0].text(0.5, 0.5, 'Sin datos de edad', ha='center', va='center')

# Edad por Churn - Boxplot
if clientes['edad'].notna().sum() > 0:
    datos_edad_churn = [
        clientes[clientes['churn']==0]['edad'].dropna(),
        clientes[clientes['churn']==1]['edad'].dropna()
    ]
    axes[0, 1].boxplot(datos_edad_churn, labels=['Activos', 'Churn'])
    axes[0, 1].set_title('Edad por Estado de Churn', fontsize=14, fontweight='bold')
    axes[0, 1].set_ylabel('Edad (años)')
else:
    axes[0, 1].text(0.5, 0.5, 'Sin datos de edad', ha='center', va='center')

# Antigüedad - Histograma
if clientes['antiguedad_meses'].notna().sum() > 0:
    axes[1, 0].hist(clientes['antiguedad_meses'].dropna(), bins=30, edgecolor='black', alpha=0.7, color='orange')
    axes[1, 0].set_title('Distribución de Antigüedad', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Antigüedad (meses)')
    axes[1, 0].set_ylabel('Frecuencia')
    axes[1, 0].axvline(clientes['antiguedad_meses'].mean(), color='red', linestyle='--', 
                       label=f'Media: {clientes["antiguedad_meses"].mean():.0f} meses')
    axes[1, 0].legend()

# Antigüedad por Churn - Boxplot
if clientes['antiguedad_meses'].notna().sum() > 0:
    datos_ant_churn = [
        clientes[clientes['churn']==0]['antiguedad_meses'].dropna(),
        clientes[clientes['churn']==1]['antiguedad_meses'].dropna()
    ]
    axes[1, 1].boxplot(datos_ant_churn, labels=['Activos', 'Churn'])
    axes[1, 1].set_title('Antigüedad por Estado de Churn', fontsize=14, fontweight='bold')
    axes[1, 1].set_ylabel('Antigüedad (meses)')

plt.tight_layout()
plt.savefig('../output/graficos/02_edad_antiguedad.png', dpi=300, bbox_inches='tight')
print("\n✓ Gráfico guardado: output/graficos/02_edad_antiguedad.png")
plt.close()

# 8. ANÁLISIS DE COMPLETITUD DE DATOS
print("\n" + "="*60)
print("COMPLETITUD DE DATOS")
print("="*60)

completitud = pd.DataFrame({
    'Columna': clientes.columns,
    'Total': len(clientes),
    'No Nulos': clientes.notna().sum(),
    'Nulos': clientes.isna().sum(),
    '% Completo': (clientes.notna().sum() / len(clientes) * 100).round(2)
}).sort_values('% Completo')

print("\n", completitud.to_string(index=False))

# 9. GRÁFICO: Completitud de datos
plt.figure(figsize=(12, 6))
completitud_sorted = completitud.sort_values('% Completo', ascending=True)
colors_completitud = ['#e74c3c' if x < 50 else '#f39c12' if x < 80 else '#2ecc71' 
                      for x in completitud_sorted['% Completo']]
plt.barh(completitud_sorted['Columna'], completitud_sorted['% Completo'], color=colors_completitud, alpha=0.8, edgecolor='black')
plt.xlabel('% de Datos Completos', fontsize=12)
plt.title('Completitud de Datos por Variable', fontsize=14, fontweight='bold')
plt.axvline(x=50, color='red', linestyle='--', linewidth=2, label='50% (Crítico)')
plt.axvline(x=80, color='orange', linestyle='--', linewidth=2, label='80% (Aceptable)')
plt.legend()
plt.tight_layout()
plt.savefig('../output/graficos/03_completitud_datos.png', dpi=300, bbox_inches='tight')
print("\n✓ Gráfico guardado: output/graficos/03_completitud_datos.png")
plt.close()

# 10. ANÁLISIS DE TRÁFICO
print("\n" + "="*60)
print("ANÁLISIS DE TRÁFICO")
print("="*60)

print("\n--- TRÁFICO JULIO 2025 ---")
print(f"Total llamadas: {len(trafico_jul):,}")
print(f"Teléfonos únicos: {trafico_jul['telefono'].nunique():,}")
print(f"Duración promedio: {trafico_jul['duracion'].mean():.2f} segundos ({trafico_jul['duracion'].mean()/60:.2f} minutos)")
print(f"Duración total: {trafico_jul['duracion'].sum()/60:,.2f} minutos")

print("\n--- TRÁFICO AGOSTO 2025 ---")
print(f"Total llamadas: {len(trafico_ago):,}")
print(f"Teléfonos únicos: {trafico_ago['telefono'].nunique():,}")
print(f"Duración promedio: {trafico_ago['duracion'].mean():.2f} segundos ({trafico_ago['duracion'].mean()/60:.2f} minutos)")

print("\n--- TRÁFICO SEPTIEMBRE 2025 ---")
print(f"Total llamadas: {len(trafico_sep):,}")
print(f"Teléfonos únicos: {trafico_sep['telefono'].nunique():,}")
print(f"Duración promedio: {trafico_sep['duracion'].mean():.2f} segundos ({trafico_sep['duracion'].mean()/60:.2f} minutos)")

# 11. GUARDAR DATASET LIMPIO
print("\n" + "="*60)
print("GUARDANDO DATASET LIMPIO")
print("="*60)

clientes.to_csv('../output/clientes_limpio.csv', index=False)
print("✓ Guardado: output/clientes_limpio.csv")

# 12. RESUMEN FINAL
print("\n" + "="*60)
print("RESUMEN DEL ANÁLISIS EXPLORATORIO")
print("="*60)

resumen = f"""
📊 CLIENTES
  • Total de clientes: {len(clientes):,}
  • Clientes activos: {(clientes['churn']==0).sum():,} ({(clientes['churn']==0).sum()/len(clientes)*100:.2f}%)
  • Clientes con churn: {clientes['churn'].sum():,} ({clientes['churn'].mean()*100:.2f}%)
  
⚠️  ALERTA: Tasa de churn muy baja ({clientes['churn'].mean()*100:.2f}%)
  → Desbalance de clases extremo
  → Aplicaremos técnicas de balanceo en el modelado

👤 DEMOGRAFÍA
  • Clientes con edad conocida: {clientes['edad'].notna().sum():,} ({clientes['edad'].notna().sum()/len(clientes)*100:.1f}%)
  • Edad promedio: {clientes['edad'].mean():.1f} años
  • Antigüedad promedio: {clientes['antiguedad_meses'].mean():.1f} meses ({clientes['antiguedad_meses'].mean()/12:.1f} años)

📞 TRÁFICO (3 meses)
  • Llamadas julio: {len(trafico_jul):,}
  • Llamadas agosto: {len(trafico_ago):,}
  • Llamadas septiembre: {len(trafico_sep):,}
  • Total llamadas: {len(trafico_jul) + len(trafico_ago) + len(trafico_sep):,}

📁 ARCHIVOS GENERADOS
  • output/clientes_limpio.csv
  • output/graficos/01_distribucion_churn.png
  • output/graficos/02_edad_antiguedad.png
  • output/graficos/03_completitud_datos.png
"""

print(resumen)

print("="*60)
print("EDA COMPLETADO ✓")
print("="*60)
print("\n🚀 Siguiente paso: Ejecutar 02_feature_engineering.py")
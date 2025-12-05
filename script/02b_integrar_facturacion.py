"""
==================================================
PROYECTO CHURN - COTEL
Script 2B: Integración de Facturación + Redefinición de Churn
==================================================
"""

import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("INTEGRACIÓN DE FACTURACIÓN Y REDEFINICIÓN DE CHURN")
print("="*60)

# 1. CARGAR DATOS
print("\n1. Cargando datos...")
df = pd.read_csv('../output/dataset_final.csv')
facturacion = pd.read_csv('../data/facturacion_6m_corregida.csv')

print(f"✓ Dataset actual: {len(df):,} registros")
print(f"✓ Facturación: {len(facturacion):,} contratos")

# Mostrar columnas de facturación para diagnóstico
print(f"\n--- Columnas en facturación ---")
print(facturacion.columns.tolist())

# Normalizar TODAS las columnas a minúsculas
facturacion.columns = facturacion.columns.str.lower()
print(f"\n✓ Columnas normalizadas a minúsculas")
print(facturacion.columns.tolist())

# 2. UNIR DATOS
print("\n" + "="*60)
print("2. UNIENDO DATOS DE FACTURACIÓN")
print("="*60)

df = df.merge(facturacion, on='contrato', how='left')
print(f"✓ Dataset unificado: {len(df):,} registros")
print(f"✓ Clientes con datos de facturación: {df['total_facturas_6m'].notna().sum():,}")

# 3. RELLENAR VALORES FALTANTES
columnas_facturas = [col for col in df.columns if any(x in col.lower() for x in 
                     ['factura', 'monto', 'deuda', 'pago', 'periodo'])]

for col in columnas_facturas:
    if df[col].dtype in ['float64', 'int64']:
        df[col] = df[col].fillna(0)

print(f"✓ Rellenadas {len(columnas_facturas)} columnas de facturación con 0")

# 4. CALCULAR MESES SIN PAGAR (CRÍTICO)
print("\n" + "="*60)
print("4. CALCULANDO MESES SIN PAGAR")
print("="*60)

# Convertir fechas
df['ultima_fecha_pago'] = pd.to_datetime(df['ultima_fecha_pago'], errors='coerce')
df['primera_factura_pendiente'] = pd.to_datetime(df['primera_factura_pendiente'], errors='coerce')

# Calcular meses desde último pago
hoy = pd.Timestamp.now()
df['meses_sin_pagar'] = 0

# Si pagó alguna vez, calcular desde última fecha
mask = df['ultima_fecha_pago'].notna()
if mask.sum() > 0:
    df.loc[mask, 'meses_sin_pagar'] = (
        (hoy - df.loc[mask, 'ultima_fecha_pago']).dt.days / 30.44
    ).round(0)

# Si nunca pagó pero tiene facturas pendientes
mask_sin_pago = (df['ultima_fecha_pago'].isna()) & (df['primera_factura_pendiente'].notna())
if mask_sin_pago.sum() > 0:
    df.loc[mask_sin_pago, 'meses_sin_pagar'] = (
        (hoy - df.loc[mask_sin_pago, 'primera_factura_pendiente']).dt.days / 30.44
    ).round(0)

# Si tiene facturas pendientes pero no fecha, asumir 3 meses
mask_pendientes = (df['facturas_pendientes'] > 0) & (df['meses_sin_pagar'] == 0)
df.loc[mask_pendientes, 'meses_sin_pagar'] = 3

print(f"✓ Meses sin pagar calculados")
print(f"  - Promedio: {df['meses_sin_pagar'].mean():.2f} meses")
print(f"  - Con >3 meses: {(df['meses_sin_pagar'] > 3).sum():,} ({(df['meses_sin_pagar'] > 3).mean()*100:.2f}%)")
print(f"  - Con >6 meses: {(df['meses_sin_pagar'] > 6).sum():,} ({(df['meses_sin_pagar'] > 6).mean()*100:.2f}%)")

# 5. RATIO DE PAGO
df['ratio_pago'] = np.where(
    df['total_facturas_6m'] > 0,
    (df['facturas_pagadas'] / df['total_facturas_6m'] * 100).round(2),
    100  # Si no tiene facturas, asumir 100% (nuevo o sin servicio)
)

print(f"\n--- Ratio de pago ---")
print(f"  - Promedio: {df['ratio_pago'].mean():.2f}%")
print(f"  - Con <50% pago: {(df['ratio_pago'] < 50).sum():,} clientes")

# 6. DEFINIR CHURN INMINENTE (NUEVA VARIABLE OBJETIVO)
print("\n" + "="*60)
print("6. DEFINIENDO CHURN INMINENTE")
print("="*60)

df['churn_inminente'] = 0

# CRITERIO 1: Cliente ya dio de baja formal
criterio1 = df['fecha_retiro'].notna()
df.loc[criterio1, 'churn_inminente'] = 1
n1 = criterio1.sum()
print(f"✓ Criterio 1 - Baja formal: {n1:,} clientes")

# CRITERIO 2: DEUDA CRÍTICA (4+ facturas pendientes O deuda > 600 Bs)
criterio2 = ((df['facturas_pendientes'] >= 4) | (df['monto_deuda'] > 600)) & (df['churn_inminente'] == 0)
df.loc[criterio2, 'churn_inminente'] = 1
n2 = criterio2.sum()
print(f"✓ Criterio 2 - Deuda crítica: {n2:,} clientes")

# CRITERIO 3: MESES SIN PAGAR (5+ meses)
criterio3 = (df['meses_sin_pagar'] >= 5) & (df['churn_inminente'] == 0)
df.loc[criterio3, 'churn_inminente'] = 1
n3 = criterio3.sum()
print(f"✓ Criterio 3 - 5+ meses sin pagar: {n3:,} clientes")

# CRITERIO 4: SIN ACTIVIDAD (últimos 2 meses)
criterio4 = ((df['llamadas_ago'] == 0) & (df['llamadas_sep'] == 0)) & (df['churn_inminente'] == 0)
df.loc[criterio4, 'churn_inminente'] = 1
n4 = criterio4.sum()
print(f"✓ Criterio 4 - Sin actividad 2 meses: {n4:,} clientes")

# CRITERIO 5: REDUCCIÓN DRÁSTICA (>80%)
criterio5 = (df['cambio_pct_llamadas'] < -80) & (df['llamadas_jul'] > 0) & (df['churn_inminente'] == 0)
df.loc[criterio5, 'churn_inminente'] = 1
n5 = criterio5.sum()
print(f"✓ Criterio 5 - Reducción drástica uso: {n5:,} clientes")

# TOTAL
total_churn = df['churn_inminente'].sum()
tasa_churn = df['churn_inminente'].mean() * 100

print(f"\n{'='*60}")
print(f"🎯 TOTAL CHURN INMINENTE: {total_churn:,} clientes ({tasa_churn:.2f}%)")
print(f"{'='*60}")

# 7. GRÁFICO: DISTRIBUCIÓN DE CHURN INMINENTE
print("\n" + "="*60)
print("7. GENERANDO GRÁFICO DE DISTRIBUCIÓN")
print("="*60)

import os
os.makedirs('../output/graficos', exist_ok=True)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Gráfico 1: Comparación Churn Original vs Churn Inminente
churn_original = df['churn'].value_counts()
churn_nuevo = df['churn_inminente'].value_counts()

x = np.arange(2)
width = 0.35

axes[0].bar(x - width/2, churn_original.values, width, label='Churn Original (0.34%)', 
            color='lightcoral', alpha=0.8, edgecolor='black')
axes[0].bar(x + width/2, churn_nuevo.values, width, label=f'Churn Inminente ({tasa_churn:.2f}%)', 
            color='crimson', alpha=0.8, edgecolor='black')

axes[0].set_xlabel('Estado', fontsize=12)
axes[0].set_ylabel('Cantidad de Clientes', fontsize=12)
axes[0].set_title('Comparación: Churn Original vs Churn Inminente', fontsize=14, fontweight='bold')
axes[0].set_xticks(x)
axes[0].set_xticklabels(['Activos (0)', 'Churn (1)'])
axes[0].legend()
axes[0].grid(axis='y', alpha=0.3)

# Gráfico 2: Distribución de Churn Inminente con porcentajes
colors = ['#2ecc71', '#e74c3c']
axes[1].bar(['Activos', 'Churn Inminente'], churn_nuevo.values, color=colors, 
            alpha=0.8, edgecolor='black')
axes[1].set_ylabel('Cantidad de Clientes', fontsize=12)
axes[1].set_title(f'Distribución Churn Inminente ({tasa_churn:.2f}%)', 
                  fontsize=14, fontweight='bold')

for i, v in enumerate(churn_nuevo.values):
    pct = (v / len(df)) * 100
    axes[1].text(i, v + max(churn_nuevo.values)*0.02, 
                f'{v:,}\n({pct:.2f}%)', 
                ha='center', fontsize=11, fontweight='bold')

axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('../output/graficos/07_churn_inminente_distribucion.png', dpi=300, bbox_inches='tight')
print("✓ Gráfico guardado: output/graficos/07_churn_inminente_distribucion.png")
plt.close()

# 8. ANÁLISIS COMPARATIVO
print("\n" + "="*60)
print("8. COMPARACIÓN: CHURN INMINENTE VS ACTIVOS")
print("="*60)

comparacion = df.groupby('churn_inminente').agg({
    'monto_deuda': 'mean',
    'facturas_pendientes': 'mean',
    'meses_sin_pagar': 'mean',
    'llamadas_total_3m': 'mean',
    'minutos_total_3m': 'mean',
    'ratio_pago': 'mean'
}).round(2)

comparacion.index = ['Activos', 'Churn Inminente']
print("\n", comparacion.to_string())

# 9. COMPLETITUD DE DATOS (SOLO VARIABLES IMPORTANTES)
print("\n" + "="*60)
print("9. COMPLETITUD DE DATOS - VARIABLES CLAVE")
print("="*60)

variables_clave = [
    'contrato', 'telefono', 'churn_inminente',
    'monto_deuda', 'facturas_pendientes', 'meses_sin_pagar', 'ratio_pago',
    'llamadas_total_3m', 'minutos_total_3m', 'tendencia_llamadas', 
    'sin_actividad', 'reduccion_uso'
]

completitud = pd.DataFrame({
    'Variable': variables_clave,
    'Total': len(df),
    'Completos': [df[col].notna().sum() for col in variables_clave],
    '% Completo': [(df[col].notna().sum() / len(df) * 100).round(2) for col in variables_clave]
})

print("\n", completitud.to_string(index=False))

# Gráfico de completitud
plt.figure(figsize=(10, 6))
plt.barh(completitud['Variable'], completitud['% Completo'], color='steelblue', alpha=0.8, edgecolor='black')
plt.xlabel('% Completitud', fontsize=12)
plt.title('Completitud de Variables Clave', fontsize=14, fontweight='bold')
plt.axvline(x=95, color='green', linestyle='--', linewidth=2, label='95% (Excelente)')
plt.legend()
plt.tight_layout()
plt.savefig('../output/graficos/08_completitud_variables_clave.png', dpi=300, bbox_inches='tight')
print("\n✓ Gráfico guardado: output/graficos/08_completitud_variables_clave.png")
plt.close()

# 10. GUARDAR DATASET FINAL
print("\n" + "="*60)
print("10. GUARDANDO DATASET FINAL")
print("="*60)

df.to_csv('../output/dataset_final_con_facturacion.csv', index=False)
print(f"✓ Guardado: output/dataset_final_con_facturacion.csv")
print(f"  Tamaño: {len(df):,} registros x {len(df.columns)} columnas")

# Guardar solo variables importantes para el modelo
df_modelo = df[variables_clave + [
    'llamadas_jul', 'llamadas_ago', 'llamadas_sep',
    'minutos_total_jul', 'minutos_total_ago', 'minutos_total_sep',
    'destinos_unicos_total_3m', 'dias_activos_total_3m',
    'cambio_pct_llamadas', 'cambio_pct_minutos', 'variabilidad_uso',
    'ciudad', 'tipo_personeria', 'valor_accion'
]].copy()

df_modelo.to_csv('../output/dataset_modelo.csv', index=False)
print(f"✓ Guardado: output/dataset_modelo.csv (solo variables importantes)")

# 11. RESUMEN FINAL
print("\n" + "="*60)
print("RESUMEN FINAL")
print("="*60)

resumen = f"""
📊 DATOS PROCESADOS
  • Total clientes: {len(df):,}
  • Clientes con facturación: {(df['total_facturas_6m'] > 0).sum():,}

🎯 CHURN INMINENTE
  • Total: {total_churn:,} clientes
  • Tasa: {tasa_churn:.2f}%
  • Desglose:
    - Baja formal: {n1:,}
    - Deuda crítica: {n2:,}
    - 5+ meses sin pagar: {n3:,}
    - Sin actividad: {n4:,}
    - Reducción drástica: {n5:,}

💰 DEUDA
  • Promedio deuda churn: Bs {comparacion.loc['Churn Inminente', 'monto_deuda']:.2f}
  • Promedio deuda activos: Bs {comparacion.loc['Activos', 'monto_deuda']:.2f}

📞 USO DEL SERVICIO
  • Llamadas promedio churn: {comparacion.loc['Churn Inminente', 'llamadas_total_3m']:.2f}
  • Llamadas promedio activos: {comparacion.loc['Activos', 'llamadas_total_3m']:.2f}

📁 ARCHIVOS GENERADOS
  • dataset_final_con_facturacion.csv (completo)
  • dataset_modelo.csv (solo variables importantes)
  • graficos/07_churn_inminente_distribucion.png
  • graficos/08_completitud_variables_clave.png
"""

print(resumen)

print("="*60)
print("INTEGRACIÓN COMPLETADA ✓")
print("="*60)
print("\n🚀 Siguiente paso: Ejecutar 03_modeling.py")
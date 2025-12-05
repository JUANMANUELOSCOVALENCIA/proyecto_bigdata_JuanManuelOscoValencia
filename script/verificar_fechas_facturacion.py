"""
==================================================
VERIFICACIÓN DE DATOS - FACTURACIÓN
Script: verificar_fechas_facturacion.py
==================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print("="*60)
print("VERIFICACIÓN DE FECHAS - FACTURACIÓN")
print("="*60)

# 1. CARGAR FACTURACIÓN
print("\n1. Cargando datos de facturación...")
facturacion = pd.read_csv('../data/facturacion_6m_corregida.csv')

print(f"✓ Total registros: {len(facturacion):,}")
print(f"✓ Contratos únicos: {facturacion['contrato'].nunique():,}")

# 2. MOSTRAR PRIMERAS FILAS
print("\n" + "="*60)
print("2. PRIMERAS 10 FILAS")
print("="*60)
print(facturacion.head(10).to_string())

# 3. COLUMNAS Y TIPOS
print("\n" + "="*60)
print("3. ESTRUCTURA DE DATOS")
print("="*60)
print(facturacion.dtypes)

# 4. CONVERTIR FECHAS
print("\n" + "="*60)
print("4. PROCESANDO FECHAS")
print("="*60)

facturacion['ultima_fecha_pago'] = pd.to_datetime(facturacion['ultima_fecha_pago'], errors='coerce')
facturacion['primera_factura_pendiente'] = pd.to_datetime(facturacion['primera_factura_pendiente'], errors='coerce')

print(f"✓ Fechas convertidas")

# 5. RANGO DE FECHAS
print("\n" + "="*60)
print("5. RANGO DE FECHAS")
print("="*60)

print(f"\n--- ÚLTIMA FECHA DE PAGO ---")
print(f"  Fecha mínima: {facturacion['ultima_fecha_pago'].min()}")
print(f"  Fecha máxima: {facturacion['ultima_fecha_pago'].max()}")
print(f"  Registros con fecha: {facturacion['ultima_fecha_pago'].notna().sum():,}")

print(f"\n--- PRIMERA FACTURA PENDIENTE ---")
print(f"  Fecha mínima: {facturacion['primera_factura_pendiente'].min()}")
print(f"  Fecha máxima: {facturacion['primera_factura_pendiente'].max()}")
print(f"  Registros con fecha: {facturacion['primera_factura_pendiente'].notna().sum():,}")

# 6. ANÁLISIS DE PERIODOS
print("\n" + "="*60)
print("6. ANÁLISIS DE PERIODOS")
print("="*60)

print(f"\n--- PERIODO_INICIO ---")
print(facturacion['periodo_inicio'].value_counts().sort_index().head(20))

print(f"\n--- PERIODO_FIN ---")
print(facturacion['periodo_fin'].value_counts().sort_index().tail(20))

# 7. ESTADÍSTICAS DE FACTURACIÓN
print("\n" + "="*60)
print("7. ESTADÍSTICAS GENERALES")
print("="*60)

stats = facturacion[['total_facturas_6m', 'facturas_pendientes', 'facturas_pagadas', 
                     'monto_total_6m', 'monto_deuda', 'monto_pagado']].describe()
print(stats.to_string())

# 8. DISTRIBUCIÓN DE ESTADOS
print("\n" + "="*60)
print("8. DISTRIBUCIÓN DE PAGOS")
print("="*60)

print(f"\n--- Facturas Pendientes ---")
print(facturacion['facturas_pendientes'].value_counts().sort_index().head(10))

print(f"\n--- Facturas Pagadas ---")
print(facturacion['facturas_pagadas'].value_counts().sort_index().head(10))

# 9. CLIENTES POR CATEGORÍA
print("\n" + "="*60)
print("9. CATEGORÍAS DE CLIENTES")
print("="*60)

sin_facturas = (facturacion['total_facturas_6m'] == 0).sum()
solo_pagadas = ((facturacion['facturas_pagadas'] > 0) & (facturacion['facturas_pendientes'] == 0)).sum()
solo_pendientes = ((facturacion['facturas_pendientes'] > 0) & (facturacion['facturas_pagadas'] == 0)).sum()
mixtas = ((facturacion['facturas_pendientes'] > 0) & (facturacion['facturas_pagadas'] > 0)).sum()

print(f"\n  Sin facturas: {sin_facturas:,}")
print(f"  Solo pagadas: {solo_pagadas:,}")
print(f"  Solo pendientes: {solo_pendientes:,}")
print(f"  Pagadas + pendientes: {mixtas:,}")

# 10. GRÁFICO DE DISTRIBUCIÓN DE PERIODOS
print("\n" + "="*60)
print("10. GENERANDO GRÁFICO")
print("="*60)

import os
os.makedirs('../output/graficos', exist_ok=True)

fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# Gráfico 1: Distribución de total facturas
axes[0, 0].hist(facturacion['total_facturas_6m'], bins=30, color='steelblue', edgecolor='black')
axes[0, 0].set_xlabel('Total Facturas (6 meses)')
axes[0, 0].set_ylabel('Cantidad de Contratos')
axes[0, 0].set_title('Distribución: Total de Facturas', fontweight='bold')
axes[0, 0].grid(alpha=0.3)

# Gráfico 2: Facturas pendientes vs pagadas
width = 0.35
x = np.arange(2)
promedios = [facturacion['facturas_pendientes'].mean(), facturacion['facturas_pagadas'].mean()]
axes[0, 1].bar(x, promedios, width, color=['crimson', 'green'], edgecolor='black')
axes[0, 1].set_ylabel('Promedio de Facturas')
axes[0, 1].set_title('Promedio: Pendientes vs Pagadas', fontweight='bold')
axes[0, 1].set_xticks(x)
axes[0, 1].set_xticklabels(['Pendientes', 'Pagadas'])
axes[0, 1].grid(axis='y', alpha=0.3)

# Gráfico 3: Distribución de deuda
facturacion_con_deuda = facturacion[facturacion['monto_deuda'] > 0]
axes[1, 0].hist(facturacion_con_deuda['monto_deuda'], bins=50, color='orange', edgecolor='black')
axes[1, 0].set_xlabel('Monto Deuda (Bs)')
axes[1, 0].set_ylabel('Cantidad de Contratos')
axes[1, 0].set_title(f'Distribución de Deuda (n={len(facturacion_con_deuda):,})', fontweight='bold')
axes[1, 0].grid(alpha=0.3)

# Gráfico 4: Categorías de clientes
categorias = ['Sin facturas', 'Solo pagadas', 'Solo pendientes', 'Mixtas']
valores = [sin_facturas, solo_pagadas, solo_pendientes, mixtas]
colors = ['gray', 'green', 'red', 'orange']
axes[1, 1].bar(categorias, valores, color=colors, edgecolor='black', alpha=0.7)
axes[1, 1].set_ylabel('Cantidad de Contratos')
axes[1, 1].set_title('Categorías de Clientes', fontweight='bold')
axes[1, 1].tick_params(axis='x', rotation=15)
axes[1, 1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('../output/graficos/verificacion_facturacion.png', dpi=300, bbox_inches='tight')
print("✓ Gráfico guardado: output/graficos/verificacion_facturacion.png")
plt.close()

print("\n" + "="*60)
print("VERIFICACIÓN COMPLETADA ✓")
print("="*60)
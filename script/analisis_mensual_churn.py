"""
==================================================
ANÁLISIS MES A MES - CHURN COTEL
Script: analisis_mensual_churn.py
==================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

print("="*60)
print("ANÁLISIS MES A MES - CHURN COTEL")
print("="*60)

# 1. CARGAR DATOS
print("\n1. Cargando datos...")

# Cargar clientes
clientes = pd.read_csv('../output/dataset_final_con_facturacion.csv')

# Cargar tráfico por mes
trafico_jul = pd.read_csv('../data/trafico_julio_corregido.csv')
trafico_ago = pd.read_csv('../data/trafico_agosto_corregido.csv')
trafico_sep = pd.read_csv('../data/trafico_sept_corregido.csv')

# Cargar facturación detallada (necesitamos el CSV ORIGINAL sin agregar)
# Si no existe, usamos el agregado
print("⚠️  Nota: Para análisis mes a mes ideal sería tener facturación DETALLADA")
print("    Trabajaremos con los datos agregados disponibles")

print(f"\n✓ Clientes: {len(clientes):,}")
print(f"✓ Tráfico julio: {len(trafico_jul):,} registros")
print(f"✓ Tráfico agosto: {len(trafico_ago):,} registros")
print(f"✓ Tráfico septiembre: {len(trafico_sep):,} registros")

# 2. PREPARAR DATOS MENSUALES
print("\n" + "="*60)
print("2. PREPARANDO ANÁLISIS MENSUAL")
print("="*60)

# Crear teléfonos únicos con tráfico por mes
telefonos_jul = set(trafico_jul['telefono'].unique())
telefonos_ago = set(trafico_ago['telefono'].unique())
telefonos_sep = set(trafico_sep['telefono'].unique())

print(f"\n✓ Teléfonos únicos con tráfico:")
print(f"  - Julio: {len(telefonos_jul):,}")
print(f"  - Agosto: {len(telefonos_ago):,}")
print(f"  - Septiembre: {len(telefonos_sep):,}")

# 3. SEGMENTACIÓN POR MES
print("\n" + "="*60)
print("3. SEGMENTACIÓN DE CLIENTES")
print("="*60)

# Preparar dataset de análisis
df = clientes[['contrato', 'telefono', 'total_facturas_6m', 'facturas_pendientes', 
               'facturas_pagadas', 'monto_deuda', 'monto_pagado', 'ratio_pago',
               'llamadas_jul', 'llamadas_ago', 'llamadas_sep',
               'minutos_total_jul', 'minutos_total_ago', 'minutos_total_sep']].copy()

# Flags de actividad
df['trafico_jul'] = df['telefono'].isin(telefonos_jul).astype(int)
df['trafico_ago'] = df['telefono'].isin(telefonos_ago).astype(int)
df['trafico_sep'] = df['telefono'].isin(telefonos_sep).astype(int)

# Flags de facturación
df['tiene_facturas'] = (df['total_facturas_6m'] > 0).astype(int)
df['tiene_deuda'] = (df['facturas_pendientes'] > 0).astype(int)
df['pago_completo'] = (df['ratio_pago'] >= 90).astype(int)

print("\n--- JULIO 2025 ---")
print(f"Con factura: {df['tiene_facturas'].sum():,}")
print(f"Con tráfico: {df['trafico_jul'].sum():,}")
print(f"Con factura + con tráfico: {((df['tiene_facturas']==1) & (df['trafico_jul']==1)).sum():,}")
print(f"Con factura + sin tráfico: {((df['tiene_facturas']==1) & (df['trafico_jul']==0)).sum():,}")
print(f"Sin factura + con tráfico: {((df['tiene_facturas']==0) & (df['trafico_jul']==1)).sum():,}")
print(f"Sin factura + sin tráfico: {((df['tiene_facturas']==0) & (df['trafico_jul']==0)).sum():,}")

print("\n--- AGOSTO 2025 ---")
print(f"Con tráfico: {df['trafico_ago'].sum():,}")
print(f"Con factura + con tráfico: {((df['tiene_facturas']==1) & (df['trafico_ago']==1)).sum():,}")
print(f"Con factura + sin tráfico: {((df['tiene_facturas']==1) & (df['trafico_ago']==0)).sum():,}")

print("\n--- SEPTIEMBRE 2025 ---")
print(f"Con tráfico: {df['trafico_sep'].sum():,}")
print(f"Con factura + con tráfico: {((df['tiene_facturas']==1) & (df['trafico_sep']==1)).sum():,}")
print(f"Con factura + sin tráfico: {((df['tiene_facturas']==1) & (df['trafico_sep']==0)).sum():,}")

# 4. DEFINIR CHURN MES A MES
print("\n" + "="*60)
print("4. DEFINICIÓN DE CHURN MES A MES")
print("="*60)

# CHURN AGOSTO = Cliente activo en julio PERO sin actividad en agosto
df['churn_agosto'] = 0
df.loc[(df['trafico_jul'] == 1) & (df['trafico_ago'] == 0), 'churn_agosto'] = 1

# CHURN SEPTIEMBRE = Cliente activo en agosto PERO sin actividad en septiembre
df['churn_sep'] = 0
df.loc[(df['trafico_ago'] == 1) & (df['trafico_sep'] == 0), 'churn_sep'] = 1

# CHURN TOTAL = Sin actividad en agosto Y septiembre (ya era parte del análisis anterior)
df['churn_total'] = 0
df.loc[(df['trafico_ago'] == 0) & (df['trafico_sep'] == 0), 'churn_total'] = 1

print(f"\n--- TASA DE CHURN POR MES ---")
print(f"Churn en Agosto: {df['churn_agosto'].sum():,} ({df['churn_agosto'].mean()*100:.2f}%)")
print(f"Churn en Septiembre: {df['churn_sep'].sum():,} ({df['churn_sep'].mean()*100:.2f}%)")
print(f"Churn Total (Ago+Sep): {df['churn_total'].sum():,} ({df['churn_total'].mean()*100:.2f}%)")

# 5. SEGMENTOS DE CLIENTES
print("\n" + "="*60)
print("5. SEGMENTOS DE CLIENTES")
print("="*60)

segmentos = pd.DataFrame({
    'Segmento': [
        'Con factura + con tráfico (3 meses)',
        'Con factura + sin tráfico (3 meses)',
        'Sin factura + con tráfico (3 meses)',
        'Sin factura + sin tráfico (3 meses)',
        '',
        'Con factura + paga completo',
        'Con factura + paga parcial',
        'Con factura + NO paga',
        '',
        'Churn en Agosto',
        'Churn en Septiembre',
        'Churn Total (2 meses sin uso)'
    ],
    'Cantidad': [
        ((df['tiene_facturas']==1) & (df['trafico_jul']==1) & (df['trafico_ago']==1) & (df['trafico_sep']==1)).sum(),
        ((df['tiene_facturas']==1) & (df['trafico_jul']==0) & (df['trafico_ago']==0) & (df['trafico_sep']==0)).sum(),
        ((df['tiene_facturas']==0) & (df['trafico_jul']==1) & (df['trafico_ago']==1) & (df['trafico_sep']==1)).sum(),
        ((df['tiene_facturas']==0) & (df['trafico_jul']==0) & (df['trafico_ago']==0) & (df['trafico_sep']==0)).sum(),
        0,
        ((df['tiene_facturas']==1) & (df['pago_completo']==1)).sum(),
        ((df['tiene_facturas']==1) & (df['tiene_deuda']==1) & (df['facturas_pagadas']>0)).sum(),
        ((df['tiene_facturas']==1) & (df['facturas_pagadas']==0)).sum(),
        0,
        df['churn_agosto'].sum(),
        df['churn_sep'].sum(),
        df['churn_total'].sum()
    ]
})

print("\n", segmentos.to_string(index=False))

# 6. ANÁLISIS DE CHURN vs PAGO
print("\n" + "="*60)
print("6. RELACIÓN: CHURN vs PAGO")
print("="*60)

# Solo clientes con facturas
df_con_facturas = df[df['tiene_facturas'] == 1].copy()

print(f"\n--- Clientes con facturas (n={len(df_con_facturas):,}) ---")

churn_pago_completo = df_con_facturas[df_con_facturas['pago_completo']==1]['churn_total'].mean() * 100
churn_pago_parcial = df_con_facturas[(df_con_facturas['pago_completo']==0) & (df_con_facturas['facturas_pagadas']>0)]['churn_total'].mean() * 100
churn_no_pago = df_con_facturas[df_con_facturas['facturas_pagadas']==0]['churn_total'].mean() * 100

print(f"Tasa de churn si paga completo (≥90%): {churn_pago_completo:.2f}%")
print(f"Tasa de churn si paga parcial: {churn_pago_parcial:.2f}%")
print(f"Tasa de churn si NO paga: {churn_no_pago:.2f}%")

# 7. LISTAS DE CLIENTES POR CATEGORÍA
print("\n" + "="*60)
print("7. EXPORTANDO LISTAS DE CLIENTES")
print("="*60)

os.makedirs('../output/segmentos', exist_ok=True)

# Lista 1: Con factura + con tráfico
lista1 = df[(df['tiene_facturas']==1) & (df['trafico_jul']==1) & (df['trafico_ago']==1) & (df['trafico_sep']==1)][
    ['contrato', 'telefono', 'total_facturas_6m', 'facturas_pendientes', 'monto_deuda', 
     'llamadas_jul', 'llamadas_ago', 'llamadas_sep', 'minutos_total_jul', 'minutos_total_ago', 'minutos_total_sep']
]
lista1.to_csv('../output/segmentos/clientes_con_factura_con_trafico.csv', index=False)
print(f"✓ Lista 1: Con factura + con tráfico ({len(lista1):,} clientes)")

# Lista 2: Con factura + sin tráfico
lista2 = df[(df['tiene_facturas']==1) & (df['trafico_jul']==0) & (df['trafico_ago']==0) & (df['trafico_sep']==0)][
    ['contrato', 'telefono', 'total_facturas_6m', 'facturas_pendientes', 'monto_deuda', 'ratio_pago']
]
lista2.to_csv('../output/segmentos/clientes_con_factura_sin_trafico.csv', index=False)
print(f"✓ Lista 2: Con factura + sin tráfico ({len(lista2):,} clientes) ⚠️ ALTO RIESGO")

# Lista 3: Sin factura + con tráfico
lista3 = df[(df['tiene_facturas']==0) & ((df['trafico_jul']==1) | (df['trafico_ago']==1) | (df['trafico_sep']==1))][
    ['contrato', 'telefono', 'llamadas_jul', 'llamadas_ago', 'llamadas_sep']
]
lista3.to_csv('../output/segmentos/clientes_sin_factura_con_trafico.csv', index=False)
print(f"✓ Lista 3: Sin factura + con tráfico ({len(lista3):,} clientes)")

# Lista 4: Clientes que NO pagan pero tienen facturas
lista4 = df[(df['tiene_facturas']==1) & (df['facturas_pagadas']==0)][
    ['contrato', 'telefono', 'facturas_pendientes', 'monto_deuda', 'trafico_jul', 'trafico_ago', 'trafico_sep']
]
lista4.to_csv('../output/segmentos/clientes_con_deuda_sin_pagar.csv', index=False)
print(f"✓ Lista 4: Con deuda sin pagar ({len(lista4):,} clientes) 🚨 CHURN INMINENTE")

# Lista 5: Churn en agosto
lista5 = df[df['churn_agosto']==1][
    ['contrato', 'telefono', 'tiene_facturas', 'facturas_pendientes', 'llamadas_jul', 'llamadas_ago']
]
lista5.to_csv('../output/segmentos/churn_agosto.csv', index=False)
print(f"✓ Lista 5: Churn en Agosto ({len(lista5):,} clientes)")

# Lista 6: Churn en septiembre
lista6 = df[df['churn_sep']==1][
    ['contrato', 'telefono', 'tiene_facturas', 'facturas_pendientes', 'llamadas_ago', 'llamadas_sep']
]
lista6.to_csv('../output/segmentos/churn_septiembre.csv', index=False)
print(f"✓ Lista 6: Churn en Septiembre ({len(lista6):,} clientes)")

# 8. GRÁFICOS
print("\n" + "="*60)
print("8. GENERANDO GRÁFICOS")
print("="*60)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Gráfico 1: Evolución de tráfico por mes
meses = ['Julio', 'Agosto', 'Septiembre']
con_trafico = [df['trafico_jul'].sum(), df['trafico_ago'].sum(), df['trafico_sep'].sum()]
axes[0, 0].plot(meses, con_trafico, marker='o', linewidth=3, markersize=10, color='steelblue')
axes[0, 0].set_ylabel('Clientes con Tráfico', fontsize=12)
axes[0, 0].set_title('Evolución: Clientes con Tráfico', fontweight='bold', fontsize=14)
axes[0, 0].grid(alpha=0.3)
for i, v in enumerate(con_trafico):
    axes[0, 0].text(i, v + 1000, f'{v:,}', ha='center', fontsize=10, fontweight='bold')

# Gráfico 2: Segmentos principales
segmentos_principales = ['Con factura\n+ tráfico', 'Con factura\nsin tráfico', 'Sin factura\n+ tráfico', 'Sin factura\nsin tráfico']
valores_segmentos = [
    ((df['tiene_facturas']==1) & (df['trafico_jul']==1) & (df['trafico_ago']==1) & (df['trafico_sep']==1)).sum(),
    ((df['tiene_facturas']==1) & (df['trafico_jul']==0) & (df['trafico_ago']==0) & (df['trafico_sep']==0)).sum(),
    ((df['tiene_facturas']==0) & ((df['trafico_jul']==1) | (df['trafico_ago']==1) | (df['trafico_sep']==1))).sum(),
    ((df['tiene_facturas']==0) & (df['trafico_jul']==0) & (df['trafico_ago']==0) & (df['trafico_sep']==0)).sum()
]
colors_seg = ['green', 'orange', 'blue', 'red']
axes[0, 1].bar(segmentos_principales, valores_segmentos, color=colors_seg, edgecolor='black', alpha=0.7)
axes[0, 1].set_ylabel('Cantidad de Clientes', fontsize=12)
axes[0, 1].set_title('Segmentos de Clientes', fontweight='bold', fontsize=14)
axes[0, 1].tick_params(axis='x', rotation=0, labelsize=9)
axes[0, 1].grid(axis='y', alpha=0.3)

# Gráfico 3: Tasa de churn por comportamiento de pago
categorias_pago = ['Paga\ncompleto\n(≥90%)', 'Paga\nparcial', 'NO paga']
tasas_churn = [churn_pago_completo, churn_pago_parcial, churn_no_pago]
colors_churn = ['green', 'orange', 'red']
axes[1, 0].bar(categorias_pago, tasas_churn, color=colors_churn, edgecolor='black', alpha=0.7)
axes[1, 0].set_ylabel('Tasa de Churn (%)', fontsize=12)
axes[1, 0].set_title('Tasa de Churn según Comportamiento de Pago', fontweight='bold', fontsize=14)
axes[1, 0].grid(axis='y', alpha=0.3)
for i, v in enumerate(tasas_churn):
    axes[1, 0].text(i, v + 1, f'{v:.1f}%', ha='center', fontsize=10, fontweight='bold')

# Gráfico 4: Churn mes a mes
meses_churn = ['Agosto', 'Septiembre', 'Total\n(2 meses)']
churn_mensual = [df['churn_agosto'].sum(), df['churn_sep'].sum(), df['churn_total'].sum()]
axes[1, 1].bar(meses_churn, churn_mensual, color=['crimson', 'darkred', 'black'], edgecolor='black', alpha=0.7)
axes[1, 1].set_ylabel('Cantidad de Clientes', fontsize=12)
axes[1, 1].set_title('Churn Mensual', fontweight='bold', fontsize=14)
axes[1, 1].grid(axis='y', alpha=0.3)
for i, v in enumerate(churn_mensual):
    axes[1, 1].text(i, v + 500, f'{v:,}', ha='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('../output/graficos/analisis_mensual_churn.png', dpi=300, bbox_inches='tight')
print("✓ Gráfico guardado: output/graficos/analisis_mensual_churn.png")
plt.close()

# 9. GUARDAR DATASET COMPLETO CON SEGMENTOS
df.to_csv('../output/dataset_con_segmentos_mensuales.csv', index=False)
print(f"\n✓ Dataset con segmentos guardado: output/dataset_con_segmentos_mensuales.csv")

# 10. RESUMEN FINAL
print("\n" + "="*60)
print("RESUMEN EJECUTIVO")
print("="*60)

print(f"""
📊 CLIENTES TOTALES: {len(df):,}

📞 ACTIVIDAD DE TRÁFICO:
  • Julio: {df['trafico_jul'].sum():,} ({df['trafico_jul'].mean()*100:.1f}%)
  • Agosto: {df['trafico_ago'].sum():,} ({df['trafico_ago'].mean()*100:.1f}%)
  • Septiembre: {df['trafico_sep'].sum():,} ({df['trafico_sep'].mean()*100:.1f}%)

💰 FACTURACIÓN:
  • Con facturas: {df['tiene_facturas'].sum():,} ({df['tiene_facturas'].mean()*100:.1f}%)
  • Pagan completo (≥90%): {df['pago_completo'].sum():,}
  • Tienen deuda: {df['tiene_deuda'].sum():,}

🎯 SEGMENTOS CLAVE:
  • Con factura + con tráfico: {lista1.shape[0]:,}
  • Con factura + sin tráfico: {lista2.shape[0]:,} ⚠️
  • Sin factura + con tráfico: {lista3.shape[0]:,}
  • Con deuda sin pagar: {lista4.shape[0]:,} 🚨

📉 CHURN:
  • Churn Agosto: {df['churn_agosto'].sum():,} ({df['churn_agosto'].mean()*100:.2f}%)
  • Churn Septiembre: {df['churn_sep'].sum():,} ({df['churn_sep'].mean()*100:.2f}%)
  • Churn Total: {df['churn_total'].sum():,} ({df['churn_total'].mean()*100:.2f}%)

🔍 INSIGHT CLAVE:
  Tasa de churn es {churn_no_pago/churn_pago_completo:.1f}x mayor 
  en clientes que NO pagan vs los que pagan completo

📁 ARCHIVOS GENERADOS:
  • 6 listas de segmentos en output/segmentos/
  • dataset_con_segmentos_mensuales.csv
  • Gráfico: analisis_mensual_churn.png
""")

print("="*60)
print("ANÁLISIS COMPLETADO ✓")
print("="*60)
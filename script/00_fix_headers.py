"""
==================================================
PROYECTO CHURN - COTEL
Script 0: Corregir Headers de CSVs
==================================================
"""

import pandas as pd

print("="*60)
print("CORRECCIÓN DE HEADERS EN CSVs")
print("="*60)

# ============================================
# 1. CLIENTES - Asignar nombres correctos
# ============================================

print("\n1. Corrigiendo Result_1.csv (CLIENTES)...")

# Leer SIN headers
clientes = pd.read_csv('../data/Result_1.csv', header=None)

# Asignar nombres de columnas correctos (según tu query SQL)
clientes.columns = [
    'contrato',
    'cod_cliente', 
    'telefono',
    'ci',
    'fecha_retiro',
    'fecha_instalacion',
    'fecha_nacimiento',
    'sexo',
    'estado_civil',
    'tipo_personeria',
    'zona',
    'ciudad',
    'ingresos',
    'rubro',
    'estado_accion',
    'valor_accion'
]

# Guardar corregido
clientes.to_csv('../data/clientes_corregido.csv', index=False)
print(f"✓ Guardado: clientes_corregido.csv ({len(clientes):,} registros)")
print(f"  Columnas: {clientes.columns.tolist()}")

# Mostrar primeras filas
print("\n--- Primeras 3 filas ---")
print(clientes.head(3))

# ============================================
# 2. TRÁFICO - Asignar nombres correctos
# ============================================

print("\n2. Corrigiendo archivos de tráfico...")

# JULIO
trafico_jul = pd.read_csv('../data/TRAFICO_TRAF2507.csv', header=None)
trafico_jul.columns = ['telefono', 'fechahora', 'duracion', 'destino', 'tipo']
trafico_jul.to_csv('../data/trafico_julio_corregido.csv', index=False)
print(f"✓ Julio: {len(trafico_jul):,} registros")

# AGOSTO
trafico_ago = pd.read_csv('../data/TRAFICO_TRAF2508.csv', header=None)
trafico_ago.columns = ['telefono', 'fechahora', 'duracion', 'destino', 'tipo']
trafico_ago.to_csv('../data/trafico_agosto_corregido.csv', index=False)
print(f"✓ Agosto: {len(trafico_ago):,} registros")

# SEPTIEMBRE
trafico_sep = pd.read_csv('../data/TRAFICO_TRAF2509.csv', header=None)
trafico_sep.columns = ['telefono', 'fechahora', 'duracion', 'destino', 'tipo']
trafico_sep.to_csv('../data/trafico_sept_corregido.csv', index=False)
print(f"✓ Septiembre: {len(trafico_sep):,} registros")

# ============================================
# 3. VERIFICACIÓN
# ============================================

print("\n" + "="*60)
print("VERIFICACIÓN DE DATOS CORREGIDOS")
print("="*60)

print("\n--- CLIENTES ---")
print(f"Total registros: {len(clientes):,}")
print(f"Columnas: {clientes.columns.tolist()}")

print("\n--- Valores únicos por columna ---")
for col in clientes.columns:
    n_unique = clientes[col].nunique()
    n_nulls = clientes[col].isna().sum()
    print(f"  {col:20s}: {n_unique:8,} únicos | {n_nulls:8,} nulls")

print("\n--- CHURN ---")
churn_count = clientes['fecha_retiro'].notna().sum()
churn_rate = (churn_count / len(clientes)) * 100
print(f"Clientes con fecha_retiro (CHURN): {churn_count:,} ({churn_rate:.2f}%)")

print("\n" + "="*60)
print("CORRECCIÓN COMPLETADA ✓")
print("="*60)
print("\nArchivos generados:")
print("  - data/clientes_corregido.csv")
print("  - data/trafico_julio_corregido.csv")
print("  - data/trafico_agosto_corregido.csv")
print("  - data/trafico_sept_corregido.csv")
print("\nAhora ejecuta: python 01_eda.py")
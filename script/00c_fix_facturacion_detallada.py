"""
==================================================
CORRECCIÓN DE HEADERS - FACTURACIÓN DETALLADA
Script: 00c_fix_facturacion_detallada.py
==================================================
"""

import pandas as pd

print("="*60)
print("CORRECCIÓN DE HEADERS - FACTURACIÓN DETALLADA")
print("="*60)

# Leer SIN headers
facturacion = pd.read_csv('../data/facturacion_detallada_6m.csv', header=None)

print(f"\n✓ Registros leídos: {len(facturacion):,}")
print(f"✓ Columnas detectadas: {len(facturacion.columns)}")

# Mostrar primeras filas para verificar
print("\n--- Primeras 5 filas (sin headers) ---")
print(facturacion.head(5))

# Asignar nombres correctos según el query SQL
facturacion.columns = [
    'CONTRATO',
    'PERIODO',
    'ESTADO',
    'MONTO_TOTAL',
    'FECHA_EMISION'
]

print("\n✓ Headers asignados correctamente")
print(f"  Columnas: {facturacion.columns.tolist()}")

# Mostrar primeras filas con headers
print("\n--- Primeras 5 filas (con headers) ---")
print(facturacion.head(5))

# Guardar corregido
facturacion.to_csv('../data/facturacion_detallada_6m_corregida.csv', index=False)

print(f"\n✓ Guardado: data/facturacion_detallada_6m_corregida.csv")
print(f"  Tamaño: {len(facturacion):,} registros")

# Estadísticas
print("\n" + "="*60)
print("ESTADÍSTICAS")
print("="*60)

print(f"\n--- Contratos únicos ---")
print(f"  Total: {facturacion['CONTRATO'].nunique():,}")

print(f"\n--- Periodos ---")
print(facturacion['PERIODO'].value_counts().sort_index())

print(f"\n--- Estados ---")
print(facturacion['ESTADO'].value_counts())

print("\n" + "="*60)
print("CORRECCIÓN COMPLETADA ✓")
print("="*60)
print("\nAhora ejecuta: python analisis_churn_mes_a_mes.py")
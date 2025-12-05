"""
==================================================
PROYECTO CHURN - COTEL
Script 0B: Corregir Headers de Facturación
==================================================
"""

import pandas as pd

print("="*60)
print("CORRECCIÓN DE HEADERS - FACTURACIÓN")
print("="*60)

# Leer SIN headers
facturacion = pd.read_csv('../data/facturacion_6m_agregada.csv', header=None)

print(f"\n✓ Registros leídos: {len(facturacion):,}")
print(f"✓ Columnas detectadas: {len(facturacion.columns)}")

# Mostrar primeras filas para verificar
print("\n--- Primeras 3 filas (sin headers) ---")
print(facturacion.head(3))

# Asignar nombres correctos según la query SQL
facturacion.columns = [
    'contrato',
    'total_facturas_6m',
    'facturas_pendientes',
    'facturas_pagadas',
    'facturas_convenio',
    'monto_total_6m',
    'monto_deuda',
    'monto_pagado',
    'monto_promedio',
    'periodo_inicio',
    'periodo_fin',
    'ultima_fecha_pago',
    'primera_factura_pendiente'
]

print("\n✓ Headers asignados correctamente")
print(f"  Columnas: {facturacion.columns.tolist()}")

# Guardar corregido
facturacion.to_csv('../data/facturacion_6m_corregida.csv', index=False)

print(f"\n✓ Guardado: data/facturacion_6m_corregida.csv")
print(f"  Tamaño: {len(facturacion):,} registros")

# Mostrar estadísticas
print("\n" + "="*60)
print("ESTADÍSTICAS")
print("="*60)

print("\n--- Primeras 5 filas con headers ---")
print(facturacion.head(5))

print("\n--- Resumen de facturación ---")
print(f"  Contratos únicos: {facturacion['contrato'].nunique():,}")
print(f"  Total facturas promedio: {facturacion['total_facturas_6m'].mean():.2f}")
print(f"  Facturas pendientes promedio: {facturacion['facturas_pendientes'].mean():.2f}")
print(f"  Deuda promedio: Bs {facturacion['monto_deuda'].mean():.2f}")
print(f"  Contratos con deuda > 0: {(facturacion['monto_deuda'] > 0).sum():,}")

print("\n" + "="*60)
print("CORRECCIÓN COMPLETADA ✓")
print("="*60)
print("\nAhora ejecuta: python 02b_integrar_facturacion.py")
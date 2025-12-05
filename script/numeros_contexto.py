"""
Obtener números para Power BI - Slide Contexto
"""
import pandas as pd

# Cargar datos
clientes = pd.read_csv('../output/powerbi/DIM_CLIENTES.csv')
trafico = pd.read_csv('../output/powerbi/FACT_TRAFICO_MENSUAL.csv')
facturacion = pd.read_csv('../output/powerbi/FACT_PAGO_MENSUAL.csv')

print("="*60)
print("NÚMEROS PARA SLIDE DE CONTEXTO")
print("="*60)

print(f"\n📊 CLIENTES:")
print(f"  Total contratos: {len(clientes):,}")

print(f"\n📞 TRÁFICO:")
print(f"  Registros agregados: {len(trafico):,}")
print(f"  Total llamadas: {trafico['cantidad_llamadas'].sum():,.0f}")
print(f"  Meses: {sorted(trafico['mes_nombre'].unique())}")
print(f"  Periodo: {trafico['mes_nombre'].min()} - {trafico['mes_nombre'].max()}")

print(f"\n💰 FACTURACIÓN:")
print(f"  Total facturas: {len(facturacion):,}")
print(f"  Monto total: Bs {facturacion['monto'].sum():,.2f}")
print(f"  Meses: {sorted(facturacion['mes_nombre'].unique())}")
print(f"  Periodo: {facturacion['mes_nombre'].min()} - {facturacion['mes_nombre'].max()}")

print(f"\n📅 RESUMEN:")
print(f"  Clientes: {len(clientes):,}")
print(f"  Llamadas: {trafico['cantidad_llamadas'].sum():,.0f}")
print(f"  Facturas: {len(facturacion):,}")
print(f"  Periodo tráfico: 3 meses (Jul-Sep 2025)")
print(f"  Periodo facturación: 6 meses (May-Oct 2025)")
"""
CSV COMPLETO PARA POWER BI - TODOS LOS DATOS
"""
import pandas as pd
import os

print("="*60)
print("CREANDO CSV COMPLETO PARA POWER BI")
print("="*60)

os.makedirs('../output/powerbi', exist_ok=True)

# Cargar datos
df = pd.read_csv('../output/dataset_churn_mes_a_mes.csv')

print(f"✓ Dataset cargado: {len(df):,} registros")

# SEGMENTACIÓN POR MES (Julio, Agosto, Septiembre)
meses_data = []

for mes_col, mes_trafico, mes_nombre in [
    ('Julio_Pago', 'Julio_Trafico', 'Julio 2025'),
    ('Agosto_Pago', 'Agosto_Trafico', 'Agosto 2025'),
    ('Sep_Pago', 'Sep_Trafico', 'Septiembre 2025')
]:
    # Contar cada segmento
    activos = ((df[mes_col] == 'PAGADO') & (df[mes_trafico] == 1)).sum()
    riesgo_medio = ((df[mes_col] == 'PAGADO') & (df[mes_trafico] == 0)).sum()
    riesgo_alto = ((df[mes_col] == 'PENDIENTE') & (df[mes_trafico] == 1)).sum()
    churn = ((df[mes_col] == 'PENDIENTE') & (df[mes_trafico] == 0)).sum()
    
    # Agregar a la lista
    meses_data.extend([
        {'Mes': mes_nombre, 'Segmento': 'Activos', 'Descripcion': 'Pagan + Usan', 'Clientes': activos, 'Color': 'Verde', 'Orden': 1},
        {'Mes': mes_nombre, 'Segmento': 'Riesgo Medio', 'Descripcion': 'Pagan + No Usan', 'Clientes': riesgo_medio, 'Color': 'Naranja', 'Orden': 2},
        {'Mes': mes_nombre, 'Segmento': 'Riesgo Alto', 'Descripcion': 'No Pagan + Usan', 'Clientes': riesgo_alto, 'Color': 'Rojo', 'Orden': 3},
        {'Mes': mes_nombre, 'Segmento': 'Churn', 'Descripcion': 'No Pagan + No Usan', 'Clientes': churn, 'Color': 'Negro', 'Orden': 4}
    ])

# Crear DataFrame
segmentacion = pd.DataFrame(meses_data)

# Calcular porcentajes por mes
for mes in segmentacion['Mes'].unique():
    mask = segmentacion['Mes'] == mes
    total = segmentacion.loc[mask, 'Clientes'].sum()
    segmentacion.loc[mask, 'Porcentaje'] = (segmentacion.loc[mask, 'Clientes'] / total * 100).round(2)

# Guardar
segmentacion.to_csv('../output/powerbi/SEGMENTACION_COMPLETA.csv', index=False, encoding='utf-8-sig')

print("\n✓ Archivo creado: SEGMENTACION_COMPLETA.csv")
print(f"  {len(segmentacion)} registros (3 meses x 4 segmentos)")

# Mostrar resumen
print("\n" + "="*60)
print("RESUMEN POR MES")
print("="*60)

for mes in ['Julio 2025', 'Agosto 2025', 'Septiembre 2025']:
    datos_mes = segmentacion[segmentacion['Mes'] == mes]
    print(f"\n{mes}:")
    print(datos_mes[['Segmento', 'Clientes', 'Porcentaje']].to_string(index=False))
    print(f"TOTAL: {datos_mes['Clientes'].sum():,} clientes")

print("\n" + "="*60)
print("ARCHIVO ÚNICO COMPLETO ✓")
print("="*60)
print("\nIMPORTA EN POWER BI:")
print("• SEGMENTACION_COMPLETA.csv")
print("• Filtra por 'Mes' para ver cada periodo")
print("• Usa 'Segmento' para las visualizaciones")
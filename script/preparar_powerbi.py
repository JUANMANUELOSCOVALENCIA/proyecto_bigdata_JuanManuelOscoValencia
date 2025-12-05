"""
==================================================
PREPARACIÓN DE DATOS PARA POWER BI
Script: preparar_powerbi.py
==================================================
"""

import pandas as pd
import numpy as np
import os

print("="*60)
print("PREPARACIÓN DE DATOS PARA POWER BI")
print("="*60)
os.makedirs('../output/powerbi', exist_ok=True)
# 1. CARGAR DATOS PRINCIPAL
print("\n1. Cargando datos...")
df = pd.read_csv('../output/dataset_churn_mes_a_mes.csv')
print(f"✓ Dataset principal: {len(df):,} registros")

# 2. CREAR TABLA DIMENSIONAL: DIM_CLIENTES
print("\n" + "="*60)
print("2. CREANDO TABLA DIMENSIONAL: DIM_CLIENTES")
print("="*60)

dim_clientes = df[[
    'contrato', 'telefono'
]].drop_duplicates().reset_index(drop=True)

dim_clientes['cliente_id'] = dim_clientes['contrato']

print(f"✓ DIM_CLIENTES: {len(dim_clientes):,} registros")
dim_clientes.to_csv('../output/powerbi/DIM_CLIENTES.csv', index=False)
print("  Guardado: output/powerbi/DIM_CLIENTES.csv")

# 3. CREAR TABLA DE HECHOS: FACT_PAGO_MENSUAL
print("\n" + "="*60)
print("3. CREANDO TABLA DE HECHOS: FACT_PAGO_MENSUAL")
print("="*60)

# Cargar facturación detallada
facturacion = pd.read_csv('../data/facturacion_detallada_6m_corregida.csv')

# Mapear estados
facturacion['PAGO'] = facturacion['ESTADO'].map({
    'P': 1, 'X': 1, 'PL': 1,  # PAGADO = 1
    'G': 0, 'NP': 0,           # NO PAGADO = 0
    'CV': 0.5                  # CONVENIO = 0.5
})

# Mapear periodo a mes nombre
periodo_mes = {
    202505: 'Mayo 2025',
    202506: 'Junio 2025',
    202507: 'Julio 2025',
    202508: 'Agosto 2025',
    202509: 'Septiembre 2025',
    202510: 'Octubre 2025'
}

facturacion['MES'] = facturacion['PERIODO'].map(periodo_mes)
facturacion['ANIO'] = 2025
facturacion['MES_NUM'] = facturacion['PERIODO'] % 100

fact_pago = facturacion[[
    'CONTRATO', 'PERIODO', 'MES', 'ANIO', 'MES_NUM',
    'ESTADO', 'PAGO', 'MONTO_TOTAL', 'FECHA_EMISION'
]].copy()

fact_pago.columns = [
    'contrato', 'periodo', 'mes_nombre', 'anio', 'mes_numero',
    'estado_pago', 'pago_realizado', 'monto', 'fecha_factura'
]

print(f"✓ FACT_PAGO_MENSUAL: {len(fact_pago):,} registros")
fact_pago.to_csv('../output/powerbi/FACT_PAGO_MENSUAL.csv', index=False)
print("  Guardado: output/powerbi/FACT_PAGO_MENSUAL.csv")

# 4. CREAR TABLA DE HECHOS: FACT_TRAFICO_MENSUAL
print("\n" + "="*60)
print("4. CREANDO TABLA DE HECHOS: FACT_TRAFICO_MENSUAL")
print("="*60)

# Tráfico julio
trafico_jul = pd.read_csv('../data/trafico_julio_corregido.csv')
trafico_jul['mes_nombre'] = 'Julio 2025'
trafico_jul['mes_numero'] = 7
trafico_jul['periodo'] = 202507

# Tráfico agosto
trafico_ago = pd.read_csv('../data/trafico_agosto_corregido.csv')
trafico_ago['mes_nombre'] = 'Agosto 2025'
trafico_ago['mes_numero'] = 8
trafico_ago['periodo'] = 202508

# Tráfico septiembre
trafico_sep = pd.read_csv('../data/trafico_sept_corregido.csv')
trafico_sep['mes_nombre'] = 'Septiembre 2025'
trafico_sep['mes_numero'] = 9
trafico_sep['periodo'] = 202509

# Combinar
trafico_all = pd.concat([trafico_jul, trafico_ago, trafico_sep], ignore_index=True)

# Agregar por teléfono y mes
fact_trafico = trafico_all.groupby(['telefono', 'periodo', 'mes_nombre', 'mes_numero']).agg({
    'duracion': ['count', 'sum', 'mean', 'max'],
    'destino': 'nunique'
}).reset_index()

fact_trafico.columns = [
    'telefono', 'periodo', 'mes_nombre', 'mes_numero',
    'cantidad_llamadas', 'minutos_totales', 'minutos_promedio', 'minutos_max',
    'destinos_unicos'
]

fact_trafico['minutos_totales'] = (fact_trafico['minutos_totales'] / 60).round(2)
fact_trafico['minutos_promedio'] = (fact_trafico['minutos_promedio'] / 60).round(2)
fact_trafico['minutos_max'] = (fact_trafico['minutos_max'] / 60).round(2)

print(f"✓ FACT_TRAFICO_MENSUAL: {len(fact_trafico):,} registros")
fact_trafico.to_csv('../output/powerbi/FACT_TRAFICO_MENSUAL.csv', index=False)
print("  Guardado: output/powerbi/FACT_TRAFICO_MENSUAL.csv")

# 5. CREAR TABLA RESUMEN: RESUMEN_MENSUAL_CLIENTE
print("\n" + "="*60)
print("5. CREANDO TABLA RESUMEN: RESUMEN_MENSUAL_CLIENTE")
print("="*60)

# Matriz de pago por cliente y mes
meses = ['Mayo', 'Junio', 'Julio', 'Agosto', 'Sep', 'Oct']
resumen = []

for mes in meses:
    col_pago = f'{mes}_Pago'
    col_trafico = f'{mes.replace("Sep", "Septiembre")}_Trafico' if mes != 'Oct' else None
    
    if col_pago in df.columns:
        temp = df[['contrato', 'telefono', col_pago]].copy()
        temp['mes'] = mes + ' 2025'
        temp['mes_numero'] = {'Mayo': 5, 'Junio': 6, 'Julio': 7, 'Agosto': 8, 'Sep': 9, 'Oct': 10}[mes]
        temp['pago_status'] = temp[col_pago]
        temp['pago_realizado'] = temp[col_pago].isin(['PAGADO']).astype(int)
        
        if col_trafico and col_trafico in df.columns:
            temp['uso_servicio'] = df[col_trafico]
        else:
            temp['uso_servicio'] = 0
        
        temp = temp[['contrato', 'telefono', 'mes', 'mes_numero', 
                     'pago_status', 'pago_realizado', 'uso_servicio']]
        resumen.append(temp)

resumen_mensual = pd.concat(resumen, ignore_index=True)

# Agregar segmento
def clasificar_segmento(row):
    if row['pago_realizado'] == 1 and row['uso_servicio'] == 1:
        return 'Activo'
    elif row['pago_realizado'] == 1 and row['uso_servicio'] == 0:
        return 'Riesgo Medio'
    elif row['pago_realizado'] == 0 and row['uso_servicio'] == 1:
        return 'Riesgo Alto'
    elif row['pago_realizado'] == 0 and row['uso_servicio'] == 0:
        return 'Churn'
    else:
        return 'Sin Datos'

resumen_mensual['segmento'] = resumen_mensual.apply(clasificar_segmento, axis=1)

print(f"✓ RESUMEN_MENSUAL_CLIENTE: {len(resumen_mensual):,} registros")
resumen_mensual.to_csv('../output/powerbi/RESUMEN_MENSUAL_CLIENTE.csv', index=False)
print("  Guardado: output/powerbi/RESUMEN_MENSUAL_CLIENTE.csv")

# 6. CREAR TABLA: DIM_FECHA
print("\n" + "="*60)
print("6. CREANDO TABLA DIMENSIONAL: DIM_FECHA")
print("="*60)

dim_fecha = pd.DataFrame({
    'periodo': [202505, 202506, 202507, 202508, 202509, 202510],
    'mes_numero': [5, 6, 7, 8, 9, 10],
    'mes_nombre': ['Mayo', 'Junio', 'Julio', 'Agosto', 'Septiembre', 'Octubre'],
    'mes_nombre_completo': ['Mayo 2025', 'Junio 2025', 'Julio 2025', 'Agosto 2025', 'Septiembre 2025', 'Octubre 2025'],
    'anio': [2025] * 6,
    'trimestre': [2, 2, 3, 3, 3, 4]
})

print(f"✓ DIM_FECHA: {len(dim_fecha):,} registros")
dim_fecha.to_csv('../output/powerbi/DIM_FECHA.csv', index=False)
print("  Guardado: output/powerbi/DIM_FECHA.csv")

# 7. RESUMEN
print("\n" + "="*60)
print("RESUMEN - ARCHIVOS PARA POWER BI")
print("="*60)

print(f"""
📊 TABLAS DIMENSIONALES:
  1. DIM_CLIENTES.csv ({len(dim_clientes):,} clientes)
  2. DIM_FECHA.csv (6 meses)

📈 TABLAS DE HECHOS:
  3. FACT_PAGO_MENSUAL.csv ({len(fact_pago):,} facturas)
  4. FACT_TRAFICO_MENSUAL.csv ({len(fact_trafico):,} registros)

📋 TABLA RESUMEN:
  5. RESUMEN_MENSUAL_CLIENTE.csv ({len(resumen_mensual):,} registros)
     → Listo para dashboards por mes

💡 MODELO ESTRELLA:
  Centro: RESUMEN_MENSUAL_CLIENTE
  Dimensiones: DIM_CLIENTES, DIM_FECHA
  Hechos: FACT_PAGO_MENSUAL, FACT_TRAFICO_MENSUAL

🎯 EN POWER BI PODRÁS:
  • Filtrar por mes (mayo-octubre)
  • Ver evolución de churn mensual
  • Segmentar por tipo de cliente
  • Analizar tráfico vs pago
  • KPIs por periodo
""")

print("="*60)
print("PREPARACIÓN COMPLETADA ✓")
print("="*60)
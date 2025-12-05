"""
==================================================
PROYECTO CHURN - COTEL
Script 4: Modelado con Separación Temporal
==================================================
Predecir churn futuro usando solo datos históricos
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix, 
                            roc_auc_score, roc_curve,
                            f1_score, recall_score, precision_score, accuracy_score)
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import warnings
import os
warnings.filterwarnings('ignore')

print("="*60)
print("MODELADO CON SEPARACIÓN TEMPORAL - COTEL")
print("="*60)

# 1. CARGAR DATOS
print("\n1. Cargando dataset...")
df = pd.read_csv('../output/dataset_final_con_facturacion.csv')
print(f"✓ Dataset: {len(df):,} registros")

# 2. DEFINIR CHURN FUTURO (usando datos de AGO/SEP)
print("\n" + "="*60)
print("2. DEFINIENDO CHURN FUTURO")
print("="*60)

# CHURN FUTURO = Sin actividad en los 2 meses siguientes (Ago + Sep)
df['churn_futuro'] = 0

# Cliente sin actividad en Ago Y Sep
criterio_sin_uso = (df['llamadas_ago'] == 0) & (df['llamadas_sep'] == 0)

# Cliente con reducción drástica (>90%)
criterio_reduccion = (df['llamadas_sep'] < df['llamadas_jul'] * 0.1) & (df['llamadas_jul'] > 0)

# Cliente con deuda crítica
criterio_deuda = (df['facturas_pendientes'] >= 4) | (df['monto_deuda'] > 600)

# Cliente con 5+ meses sin pagar
criterio_sin_pagar = df['meses_sin_pagar'] >= 5

# Baja formal
criterio_baja = df['fecha_retiro'].notna()

# Combinar criterios
df.loc[criterio_sin_uso | criterio_reduccion | criterio_deuda | criterio_sin_pagar | criterio_baja, 'churn_futuro'] = 1

print(f"\n✓ Churn futuro definido")
print(f"  Total: {df['churn_futuro'].sum():,} clientes ({df['churn_futuro'].mean()*100:.2f}%)")

# 3. FEATURES HISTÓRICAS (solo JULIO + datos de facturación)
print("\n" + "="*60)
print("3. SELECCIÓN DE FEATURES HISTÓRICAS")
print("="*60)

features_historicas = [
    # TRÁFICO JULIO (último mes conocido)
    'llamadas_jul',
    'minutos_total_jul',
    'destinos_unicos_jul',
    'dias_activos_jul',
    'duracion_promedio_jul',
    'duracion_std_jul',
    'duracion_max_jul',
    
    # FACTURACIÓN (histórica, no incluye futuro)
    'total_facturas_6m',
    'facturas_pendientes',
    'facturas_pagadas',
    'facturas_convenio',
    'monto_total_6m',
    'monto_deuda',
    'monto_pagado',
    'monto_promedio',
    'meses_sin_pagar',
    'ratio_pago',
    
    # SERVICIO
    'antiguedad_meses',
    'valor_accion',
    'tipo_personeria_encoded'
]

print(f"\n✓ Features históricas: {len(features_historicas)}")

# 4. PREPARACIÓN
print("\n" + "="*60)
print("4. PREPARACIÓN DE DATOS")
print("="*60)

# Encoding
le = LabelEncoder()
df['tipo_personeria_encoded'] = le.fit_transform(df['tipo_personeria'].astype(str))

# Rellenar NaNs
for col in features_historicas:
    if col in df.columns:
        df[col] = df[col].fillna(0)

# Verificar que las columnas existen
features_disponibles = [f for f in features_historicas if f in df.columns]
print(f"\n✓ Features disponibles: {len(features_disponibles)}/{len(features_historicas)}")

X = df[features_disponibles].copy()
y = df['churn_futuro'].copy()

print(f"\n✓ X shape: {X.shape}")
print(f"✓ y distribución:")
print(y.value_counts())
print(f"  Tasa churn: {y.mean()*100:.2f}%")

# 5. TRAIN-TEST SPLIT
print("\n" + "="*60)
print("5. SPLIT DE DATOS")
print("="*60)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n✓ Train: {len(X_train):,}")
print(f"✓ Test: {len(X_test):,}")
print(f"✓ Train churn: {y_train.mean()*100:.2f}%")
print(f"✓ Test churn: {y_test.mean()*100:.2f}%")

# 6. ESCALADO
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\n✓ Features escaladas")

# 7. BALANCEO CON SMOTE
print("\n" + "="*60)
print("7. BALANCEO DE CLASES (SMOTE)")
print("="*60)

print(f"\nAntes de SMOTE:")
print(f"  Clase 0: {(y_train == 0).sum():,}")
print(f"  Clase 1: {(y_train == 1).sum():,}")

smote = SMOTE(random_state=42, sampling_strategy=0.5)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)

print(f"\nDespués de SMOTE:")
print(f"  Clase 0: {(y_train_balanced == 0).sum():,}")
print(f"  Clase 1: {(y_train_balanced == 1).sum():,}")

# 8. MODELADO
print("\n" + "="*60)
print("8. ENTRENAMIENTO DE MODELOS")
print("="*60)

modelos = {}
resultados = {}

# 8.1 LOGISTIC REGRESSION
print("\n--- 8.1 REGRESIÓN LOGÍSTICA ---")
lr = LogisticRegression(random_state=42, max_iter=1000)
lr.fit(X_train_balanced, y_train_balanced)

y_pred_lr = lr.predict(X_test_scaled)
y_proba_lr = lr.predict_proba(X_test_scaled)[:, 1]

print(f"✓ Entrenado")
print(f"  Accuracy: {accuracy_score(y_test, y_pred_lr):.4f}")
print(f"  Recall: {recall_score(y_test, y_pred_lr):.4f}")
print(f"  Precision: {precision_score(y_test, y_pred_lr):.4f}")
print(f"  F1-Score: {f1_score(y_test, y_pred_lr):.4f}")
print(f"  ROC-AUC: {roc_auc_score(y_test, y_proba_lr):.4f}")

resultados['Logistic Regression'] = {
    'accuracy': accuracy_score(y_test, y_pred_lr),
    'recall': recall_score(y_test, y_pred_lr),
    'precision': precision_score(y_test, y_pred_lr),
    'f1': f1_score(y_test, y_pred_lr),
    'roc_auc': roc_auc_score(y_test, y_proba_lr),
    'y_pred': y_pred_lr,
    'y_proba': y_proba_lr
}
modelos['Logistic Regression'] = lr

# 8.2 RANDOM FOREST
print("\n--- 8.2 RANDOM FOREST ---")
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=15,
    min_samples_split=10,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train_balanced, y_train_balanced)

y_pred_rf = rf.predict(X_test_scaled)
y_proba_rf = rf.predict_proba(X_test_scaled)[:, 1]

print(f"✓ Entrenado")
print(f"  Accuracy: {accuracy_score(y_test, y_pred_rf):.4f}")
print(f"  Recall: {recall_score(y_test, y_pred_rf):.4f}")
print(f"  Precision: {precision_score(y_test, y_pred_rf):.4f}")
print(f"  F1-Score: {f1_score(y_test, y_pred_rf):.4f}")
print(f"  ROC-AUC: {roc_auc_score(y_test, y_proba_rf):.4f}")

resultados['Random Forest'] = {
    'accuracy': accuracy_score(y_test, y_pred_rf),
    'recall': recall_score(y_test, y_pred_rf),
    'precision': precision_score(y_test, y_pred_rf),
    'f1': f1_score(y_test, y_pred_rf),
    'roc_auc': roc_auc_score(y_test, y_proba_rf),
    'y_pred': y_pred_rf,
    'y_proba': y_proba_rf
}
modelos['Random Forest'] = rf

# 8.3 XGBOOST
print("\n--- 8.3 XGBOOST ---")
xgb_model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=8,
    learning_rate=0.1,
    random_state=42,
    eval_metric='logloss'
)
xgb_model.fit(X_train_balanced, y_train_balanced)

y_pred_xgb = xgb_model.predict(X_test_scaled)
y_proba_xgb = xgb_model.predict_proba(X_test_scaled)[:, 1]

print(f"✓ Entrenado")
print(f"  Accuracy: {accuracy_score(y_test, y_pred_xgb):.4f}")
print(f"  Recall: {recall_score(y_test, y_pred_xgb):.4f}")
print(f"  Precision: {precision_score(y_test, y_pred_xgb):.4f}")
print(f"  F1-Score: {f1_score(y_test, y_pred_xgb):.4f}")
print(f"  ROC-AUC: {roc_auc_score(y_test, y_proba_xgb):.4f}")

resultados['XGBoost'] = {
    'accuracy': accuracy_score(y_test, y_pred_xgb),
    'recall': recall_score(y_test, y_pred_xgb),
    'precision': precision_score(y_test, y_pred_xgb),
    'f1': f1_score(y_test, y_pred_xgb),
    'roc_auc': roc_auc_score(y_test, y_proba_xgb),
    'y_pred': y_pred_xgb,
    'y_proba': y_proba_xgb
}
modelos['XGBoost'] = xgb_model

# 9. COMPARACIÓN
print("\n" + "="*60)
print("9. COMPARACIÓN DE MODELOS")
print("="*60)

comparacion = pd.DataFrame(resultados).T
comparacion = comparacion[['accuracy', 'recall', 'precision', 'f1', 'roc_auc']]
print("\n", comparacion.to_string())

mejor_modelo_nombre = comparacion['f1'].idxmax()
mejor_modelo = modelos[mejor_modelo_nombre]

print(f"\n🏆 MEJOR MODELO: {mejor_modelo_nombre}")
print(f"   F1-Score: {comparacion.loc[mejor_modelo_nombre, 'f1']:.4f}")
print(f"   ROC-AUC: {comparacion.loc[mejor_modelo_nombre, 'roc_auc']:.4f}")

# 10. PREDICCIONES FINALES
print("\n" + "="*60)
print("10. PREDICCIONES FINALES")
print("="*60)

X_all_scaled = scaler.transform(X)
df['churn_predicho'] = mejor_modelo.predict(X_all_scaled)
df['probabilidad_churn'] = mejor_modelo.predict_proba(X_all_scaled)[:, 1]

df['riesgo_churn'] = pd.cut(df['probabilidad_churn'], 
                             bins=[0, 0.3, 0.7, 1.0],
                             labels=['Bajo', 'Medio', 'Alto'])

print(f"\n✓ Predicciones completadas")
print(f"\n--- Distribución de Riesgo ---")
print(df['riesgo_churn'].value_counts())

# 11. GUARDAR
print("\n" + "="*60)
print("11. GUARDANDO RESULTADOS")
print("="*60)

df.to_csv('../output/predicciones_temporales.csv', index=False)
print(f"✓ Guardado: predicciones_temporales.csv")

alto_riesgo = df[df['riesgo_churn'] == 'Alto'].sort_values('probabilidad_churn', ascending=False)
alto_riesgo[['contrato', 'telefono', 'churn_futuro', 'churn_predicho', 'probabilidad_churn',
             'llamadas_jul', 'minutos_total_jul', 'monto_deuda', 'facturas_pendientes']].to_csv(
    '../output/clientes_alto_riesgo_temporal.csv', index=False
)
print(f"✓ Guardado: clientes_alto_riesgo_temporal.csv ({len(alto_riesgo):,} clientes)")

# 12. RESUMEN
print("\n" + "="*60)
print("RESUMEN FINAL")
print("="*60)

print(f"""
🎯 MEJOR MODELO: {mejor_modelo_nombre}
  • F1-Score: {comparacion.loc[mejor_modelo_nombre, 'f1']:.4f}
  • ROC-AUC: {comparacion.loc[mejor_modelo_nombre, 'roc_auc']:.4f}
  • Recall: {comparacion.loc[mejor_modelo_nombre, 'recall']:.4f}
  • Precision: {comparacion.loc[mejor_modelo_nombre, 'precision']:.4f}

⚠️  CLIENTES EN RIESGO
  • Alto: {(df['riesgo_churn']=='Alto').sum():,}
  • Medio: {(df['riesgo_churn']=='Medio').sum():,}
  • Bajo: {(df['riesgo_churn']=='Bajo').sum():,}

✅ PROYECTO COMPLETADO
""")

print("="*60)
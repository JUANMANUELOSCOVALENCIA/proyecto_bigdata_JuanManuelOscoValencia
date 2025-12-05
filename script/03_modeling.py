"""
==================================================
PROYECTO CHURN - COTEL
Script 3: Modelado y Predicción de Churn Inminente
==================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix, 
                            roc_auc_score, roc_curve, precision_recall_curve,
                            f1_score, recall_score, precision_score, accuracy_score)
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import warnings
import os
warnings.filterwarnings('ignore')

print("="*60)
print("MODELADO DE CHURN INMINENTE - COTEL")
print("="*60)

# 1. CARGAR DATOS
print("\n1. Cargando dataset...")
df = pd.read_csv('../output/dataset_final_con_facturacion.csv')
print(f"✓ Dataset: {len(df):,} registros x {len(df.columns)} columnas")
print(f"✓ Tasa de churn inminente: {df['churn_inminente'].mean()*100:.2f}%")

# 2. SELECCIÓN DE FEATURES
print("\n" + "="*60)
print("2. SELECCIÓN DE FEATURES")
print("="*60)

# Features numéricas para el modelo
features_numericas = [
    # TRÁFICO - Por mes
    'llamadas_jul', 'llamadas_ago', 'llamadas_sep',
    'minutos_total_jul', 'minutos_total_ago', 'minutos_total_sep',
    'destinos_unicos_jul', 'destinos_unicos_ago', 'destinos_unicos_sep',
    'dias_activos_jul', 'dias_activos_ago', 'dias_activos_sep',
    
    # TRÁFICO - Agregados 3 meses
    'llamadas_total_3m', 'minutos_total_3m', 'destinos_unicos_total_3m',
    'dias_activos_total_3m', 'llamadas_promedio_mes', 'minutos_promedio_mes',
    'minutos_por_llamada', 'llamadas_por_dia',
    
    # TENDENCIAS DE USO
    'tendencia_llamadas', 'cambio_pct_llamadas', 
    'tendencia_minutos', 'cambio_pct_minutos',
    'variabilidad_uso',
    
    # FACTURACIÓN Y DEUDA
    'total_facturas_6m', 'facturas_pendientes', 'facturas_pagadas',
    'monto_total_6m', 'monto_deuda', 'monto_pagado',
    'meses_sin_pagar', 'ratio_pago',
    
    # DEMOGRÁFICAS Y SERVICIO
    'antiguedad_meses', 'valor_accion'
]

# Features categóricas
features_categoricas = [
    'tipo_personeria'
]

print(f"\n✓ Features numéricas: {len(features_numericas)}")
print(f"✓ Features categóricas: {len(features_categoricas)}")

# 3. PREPARACIÓN DE DATOS
print("\n" + "="*60)
print("3. PREPARACIÓN DE DATOS")
print("="*60)

# Crear copia para modelado
df_model = df.copy()

# Rellenar NaNs en features numéricas con 0
for col in features_numericas:
    if col in df_model.columns:
        df_model[col] = df_model[col].fillna(0)

# Encoding de variables categóricas
le_dict = {}
features_finales = features_numericas.copy()

for col in features_categoricas:
    if col in df_model.columns:
        le = LabelEncoder()
        col_encoded = col + '_encoded'
        df_model[col_encoded] = le.fit_transform(df_model[col].astype(str))
        le_dict[col] = le
        features_finales.append(col_encoded)

print(f"\n✓ Variables categóricas codificadas")

# Preparar X e y
X = df_model[features_finales].copy()
y = df_model['churn_inminente'].copy()

print(f"\n✓ X shape: {X.shape}")
print(f"✓ y shape: {y.shape}")
print(f"✓ Distribución y:")
print(y.value_counts())
print(f"  Tasa churn: {y.mean()*100:.2f}%")

# 4. TRAIN-TEST SPLIT
print("\n" + "="*60)
print("4. SPLIT DE DATOS (80/20)")
print("="*60)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n✓ Train: {len(X_train):,} registros")
print(f"✓ Test: {len(X_test):,} registros")
print(f"✓ Train churn rate: {y_train.mean()*100:.2f}%")
print(f"✓ Test churn rate: {y_test.mean()*100:.2f}%")

# 5. ESCALADO
print("\n" + "="*60)
print("5. ESCALADO DE FEATURES")
print("="*60)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("✓ Features escaladas (StandardScaler)")

# 6. MODELADO
print("\n" + "="*60)
print("6. ENTRENAMIENTO DE MODELOS")
print("="*60)

modelos = {}
resultados = {}

# 6.1 REGRESIÓN LOGÍSTICA (Baseline)
print("\n--- 6.1 REGRESIÓN LOGÍSTICA (Baseline) ---")
lr = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
lr.fit(X_train_scaled, y_train)
modelos['Logistic Regression'] = lr

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

# 6.2 RANDOM FOREST
print("\n--- 6.2 RANDOM FOREST ---")
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=15,
    min_samples_split=10,
    min_samples_leaf=5,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train_scaled, y_train)
modelos['Random Forest'] = rf

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

# 6.3 XGBOOST
print("\n--- 6.3 XGBOOST ---")

# Calcular scale_pos_weight para balanceo
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

xgb_model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=8,
    learning_rate=0.1,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    eval_metric='logloss',
    use_label_encoder=False
)
xgb_model.fit(X_train_scaled, y_train)
modelos['XGBoost'] = xgb_model

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

# 7. COMPARACIÓN DE MODELOS
print("\n" + "="*60)
print("7. COMPARACIÓN DE MODELOS")
print("="*60)

comparacion = pd.DataFrame(resultados).T
comparacion = comparacion[['accuracy', 'recall', 'precision', 'f1', 'roc_auc']]
print("\n", comparacion.to_string())

# Identificar mejor modelo
mejor_modelo_nombre = comparacion['f1'].idxmax()
mejor_modelo = modelos[mejor_modelo_nombre]
print(f"\n🏆 MEJOR MODELO: {mejor_modelo_nombre}")
print(f"   F1-Score: {comparacion.loc[mejor_modelo_nombre, 'f1']:.4f}")
print(f"   ROC-AUC: {comparacion.loc[mejor_modelo_nombre, 'roc_auc']:.4f}")

# 8. GRÁFICOS
print("\n" + "="*60)
print("8. GENERANDO GRÁFICOS")
print("="*60)

os.makedirs('../output/graficos', exist_ok=True)

# 8.1 Matriz de Confusión (3 modelos)
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, (nombre, res) in enumerate(resultados.items()):
    cm = confusion_matrix(y_test, res['y_pred'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx], cbar=False)
    axes[idx].set_title(f'{nombre}\nF1={res["f1"]:.3f}', fontweight='bold', fontsize=12)
    axes[idx].set_ylabel('Real', fontsize=10)
    axes[idx].set_xlabel('Predicho', fontsize=10)
    axes[idx].set_xticklabels(['Activo', 'Churn'])
    axes[idx].set_yticklabels(['Activo', 'Churn'])

plt.tight_layout()
plt.savefig('../output/graficos/09_confusion_matrices.png', dpi=300, bbox_inches='tight')
print("✓ Guardado: output/graficos/09_confusion_matrices.png")
plt.close()

# 8.2 Curvas ROC
plt.figure(figsize=(10, 8))

colors = ['#3498db', '#e74c3c', '#2ecc71']
for idx, (nombre, res) in enumerate(resultados.items()):
    fpr, tpr, _ = roc_curve(y_test, res['y_proba'])
    plt.plot(fpr, tpr, label=f'{nombre} (AUC={res["roc_auc"]:.3f})', 
             linewidth=2.5, color=colors[idx])

plt.plot([0, 1], [0, 1], 'k--', label='Random', linewidth=1.5)
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('Curvas ROC - Comparación de Modelos', fontsize=14, fontweight='bold')
plt.legend(fontsize=11, loc='lower right')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('../output/graficos/10_roc_curves.png', dpi=300, bbox_inches='tight')
print("✓ Guardado: output/graficos/10_roc_curves.png")
plt.close()

# 8.3 Feature Importance (Random Forest y XGBoost)
fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# Random Forest
importances_rf = rf.feature_importances_
indices_rf = np.argsort(importances_rf)[-15:]  # Top 15
axes[0].barh(range(len(indices_rf)), importances_rf[indices_rf], color='steelblue', alpha=0.8, edgecolor='black')
axes[0].set_yticks(range(len(indices_rf)))
axes[0].set_yticklabels([X.columns[i] for i in indices_rf], fontsize=9)
axes[0].set_xlabel('Importancia', fontsize=11)
axes[0].set_title('Top 15 Features - Random Forest', fontweight='bold', fontsize=12)
axes[0].grid(axis='x', alpha=0.3)

# XGBoost
importances_xgb = xgb_model.feature_importances_
indices_xgb = np.argsort(importances_xgb)[-15:]  # Top 15
axes[1].barh(range(len(indices_xgb)), importances_xgb[indices_xgb], color='orange', alpha=0.8, edgecolor='black')
axes[1].set_yticks(range(len(indices_xgb)))
axes[1].set_yticklabels([X.columns[i] for i in indices_xgb], fontsize=9)
axes[1].set_xlabel('Importancia', fontsize=11)
axes[1].set_title('Top 15 Features - XGBoost', fontweight='bold', fontsize=12)
axes[1].grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('../output/graficos/11_feature_importance.png', dpi=300, bbox_inches='tight')
print("✓ Guardado: output/graficos/11_feature_importance.png")
plt.close()

# 8.4 Gráfico de Comparación de Métricas
fig, ax = plt.subplots(figsize=(12, 6))

metricas = ['accuracy', 'recall', 'precision', 'f1', 'roc_auc']
x = np.arange(len(metricas))
width = 0.25

for idx, (nombre, color) in enumerate(zip(resultados.keys(), colors)):
    valores = [resultados[nombre][m] for m in metricas]
    ax.bar(x + idx*width, valores, width, label=nombre, color=color, alpha=0.8, edgecolor='black')

ax.set_xlabel('Métricas', fontsize=12, fontweight='bold')
ax.set_ylabel('Score', fontsize=12, fontweight='bold')
ax.set_title('Comparación de Métricas - Todos los Modelos', fontsize=14, fontweight='bold')
ax.set_xticks(x + width)
ax.set_xticklabels([m.upper() for m in metricas], fontsize=10)
ax.legend(fontsize=10)
ax.set_ylim(0, 1.1)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('../output/graficos/12_comparacion_metricas.png', dpi=300, bbox_inches='tight')
print("✓ Guardado: output/graficos/12_comparacion_metricas.png")
plt.close()

# 9. PREDICCIONES EN TODO EL DATASET
print("\n" + "="*60)
print("9. PREDICCIONES FINALES CON MEJOR MODELO")
print("="*60)

# Escalar todo el dataset
X_all_scaled = scaler.transform(X)

# Predecir con mejor modelo
df['churn_predicho'] = mejor_modelo.predict(X_all_scaled)
df['probabilidad_churn'] = mejor_modelo.predict_proba(X_all_scaled)[:, 1]

# Clasificar riesgo
df['riesgo_churn'] = pd.cut(df['probabilidad_churn'], 
                             bins=[0, 0.5, 0.8, 1.0],
                             labels=['Bajo', 'Medio', 'Alto'])

print(f"\n✓ Predicciones completadas para {len(df):,} clientes")
print(f"\n--- Distribución de Riesgo ---")
print(df['riesgo_churn'].value_counts())
print(f"\n--- Distribución de Predicción vs Real ---")
print(pd.crosstab(df['churn_inminente'], df['churn_predicho'], 
                  rownames=['Real'], colnames=['Predicho'], margins=True))

# 10. GUARDAR RESULTADOS
print("\n" + "="*60)
print("10. GUARDANDO RESULTADOS")
print("="*60)

# Guardar predicciones completas
df.to_csv('../output/predicciones_completas.csv', index=False)
print(f"✓ Guardado: output/predicciones_completas.csv")

# Guardar solo clientes de ALTO RIESGO
alto_riesgo = df[df['riesgo_churn'] == 'Alto'].sort_values('probabilidad_churn', ascending=False)
alto_riesgo_export = alto_riesgo[[
    'contrato', 'telefono', 'churn_inminente', 'churn_predicho', 'probabilidad_churn',
    'llamadas_total_3m', 'minutos_total_3m', 'monto_deuda', 'facturas_pendientes',
    'meses_sin_pagar', 'sin_actividad', 'reduccion_uso'
]].copy()

alto_riesgo_export.to_csv('../output/clientes_alto_riesgo.csv', index=False)
print(f"✓ Guardado: output/clientes_alto_riesgo.csv ({len(alto_riesgo):,} clientes)")

# Guardar tabla de comparación de modelos
comparacion.to_csv('../output/comparacion_modelos.csv')
print(f"✓ Guardado: output/comparacion_modelos.csv")

# 11. RESUMEN FINAL
print("\n" + "="*60)
print("RESUMEN FINAL - PROYECTO CHURN COTEL")
print("="*60)

resumen = f"""
📊 DATASET
  • Total clientes: {len(df):,}
  • Features utilizadas: {len(features_finales)}
  • Train/Test split: 80/20

🎯 CHURN INMINENTE (Variable Objetivo)
  • Tasa real: {df['churn_inminente'].mean()*100:.2f}%
  • Total casos: {df['churn_inminente'].sum():,} clientes

🏆 MEJOR MODELO: {mejor_modelo_nombre}
  • Accuracy: {comparacion.loc[mejor_modelo_nombre, 'accuracy']:.4f}
  • Recall: {comparacion.loc[mejor_modelo_nombre, 'recall']:.4f}
  • Precision: {comparacion.loc[mejor_modelo_nombre, 'precision']:.4f}
  • F1-Score: {comparacion.loc[mejor_modelo_nombre, 'f1']:.4f}
  • ROC-AUC: {comparacion.loc[mejor_modelo_nombre, 'roc_auc']:.4f}

⚠️  CLIENTES EN RIESGO
  • Riesgo Alto: {(df['riesgo_churn']=='Alto').sum():,} clientes
  • Riesgo Medio: {(df['riesgo_churn']=='Medio').sum():,} clientes
  • Riesgo Bajo: {(df['riesgo_churn']=='Bajo').sum():,} clientes

📁 ARCHIVOS GENERADOS
  • predicciones_completas.csv (todos los clientes)
  • clientes_alto_riesgo.csv (para retención prioritaria)
  • comparacion_modelos.csv
  • 4 gráficos en output/graficos/

🎉 PROYECTO COMPLETADO EXITOSAMENTE
"""

print(resumen)

print("="*60)
print("MODELADO COMPLETADO ✓")
print("="*60)
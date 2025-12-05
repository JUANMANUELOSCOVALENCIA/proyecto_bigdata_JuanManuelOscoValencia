import pandas as pd
import os

os.makedirs('../output/powerbi', exist_ok=True)

# TABLA SIMPLE CON LOS MESES
meses = pd.DataFrame({
    'Tipo': ['Trafico', 'Facturacion'],
    'Meses': [3, 6]
})

meses.to_csv('../output/powerbi/MESES_DATOS.csv', index=False, encoding='utf-8-sig')

print("✓ Archivo creado: MESES_DATOS.csv")
print("\n", meses)
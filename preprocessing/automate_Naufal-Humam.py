from sklearn.preprocessing import StandardScaler
import pandas as pd
from IPython.display import display
import os

# # Mengambil base directory dari file skrip ini berada (yaitu folder 'preprocessing')
# base_dir = os.path.dirname(os.path.abspath(__file__))

# # Menggabungkan path ke file creditcard.csv yang berada di luar folder preprocessing
# csv_path = os.path.join(base_dir, 'creditcard.csv')
# 1. Ambil path absolut dengan dinamis

base_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(base_dir, '../creditcard.csv')

print(f"Mencoba membaca file dari: {csv_path}")

df = pd.read_csv(csv_path)
df.head()

# Scale 'Amount' and 'Time' features
scaler = StandardScaler()
df['Amount'] = scaler.fit_transform(df[['Amount']])
df['Time'] = scaler.fit_transform(df[['Time']])

print("Scaled 'Time' and 'Amount' features. Displaying head to confirm:")
display(df[['Time', 'Amount', 'Class']].head())
        

df.to_csv('creditcard_preprocessing.csv', index=False)

print('Preprocessed data exported to creditcard_preprocessing.csv')

# Display the first few rows of the exported data to confirm
exported_df = pd.read_csv('creditcard_preprocessing.csv')

display(exported_df.head())

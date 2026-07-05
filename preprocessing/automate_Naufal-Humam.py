from sklearn.preprocessing import StandardScaler
import pandas as pd
from IPython.display import display
import os
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

# # Mengambil base directory dari file skrip ini berada (yaitu folder 'preprocessing')
# base_dir = os.path.dirname(os.path.abspath(__file__))

# # Menggabungkan path ke file creditcard.csv yang berada di luar folder preprocessing
# csv_path = os.path.join(base_dir, 'creditcard.csv')
# 1. Ambil path absolut dengan dinamis

base_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(base_dir, '../creditcard_raw.csv')

print(f"Mencoba membaca file dari: {csv_path}")

df = pd.read_csv(csv_path)
df.head()

# Scale 'Amount' and 'Time' features
scaler = StandardScaler()
df['Amount'] = scaler.fit_transform(df[['Amount']])
df['Time'] = scaler.fit_transform(df[['Time']])

print("Scaled 'Time' and 'Amount' features. Displaying head to confirm:")
display(df[['Time', 'Amount', 'Class']].head())

# Penambahan pembagian data preprocessing dan SMOTE

# dataset_path = sys.argv[1] if len(sys.argv) > 1 else "../creditcardfraud/creditcard.csv"

# df = pd.read_csv(dataset_path)
# 1. Pisahkan Fitur dan Target
X = df.drop(columns=['Class'])
y = df['Class']

# 2. Split Data (80% Train, 20% Test) - WAJIB SEBELUM SMOTE
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 3. Oversampling dengan SMOTE (Hanya pada Data Train)
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# Gabungkan X dan y
train_resampled = X_train_res.copy()
train_resampled["target"] = y_train_res
test_resampled = X_test.copy()
test_resampled["target"] = y_test

# Simpan ke CSV
train_resampled.to_csv("creditcard_preprocessing_train.csv", index=False)
test_resampled.to_csv("creditcard_preprocessing_test.csv", index=False)

print('Preprocessed data exported to creditcard_preprocessing_train.csv')
print('Preprocessed data exported to creditcard_preprocessing_test.csv')

# Display the first few rows of the exported data to confirm
exported_df_train = pd.read_csv('creditcard_preprocessing_train.csv')
exported_df_test = pd.read_csv('creditcard_preprocessing_test.csv')

display(exported_df_train.head())
display(exported_df_test.head())

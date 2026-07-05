from sklearn.preprocessing import StandardScaler
import pandas as pd
from IPython.display import display

df = pd.read_csv('../creditcard.csv')
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

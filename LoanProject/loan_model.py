# Необхідні імпорти
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
# Завантажуємо дані та робимо базовий аналіз і підготовку даних
df = pd.read_csv('loan_data.csv')
print(df.head().to_string())
print(df.describe().round(2).to_string(), '\n') # Розподіл даних
print(df.isna().sum(), '\n') # Відсутні значення
# Заповнюємо відсутні значення
for i in ['Gender', 'Dependents', 'Self_Employed', 'Credit_History']:
    df[i] = df[i].fillna(df[i].mode()[0])

df['Loan_Amount_Term'] = df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].median())
print(df.isna().sum(), '\n')

# Кореляція
num_df = df.select_dtypes(include='number')
print(num_df.corr().round(2).to_string(), '\n')

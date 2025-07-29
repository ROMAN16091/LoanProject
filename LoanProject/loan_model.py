# Необхідні імпорти
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
# Завантажуємо дані та робимо базовий аналіз і підготовку даних
df = pd.read_csv('loan_data.csv')
print(df.head().to_string())
print('Розподіл даних:\n', df.describe().round(2).to_string(), '\n') # Розподіл даних
print('Відсутні значення:\n',df.isna().sum(), '\n') # Відсутні значення
# Заповнюємо відсутні значення
for i in ['Gender', 'Dependents', 'Self_Employed', 'Credit_History']:
    df[i] = df[i].fillna(df[i].mode()[0])

df['Loan_Amount_Term'] = df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].median())
print('Відсутні значення:\n',df.isna().sum(), '\n')

# Позбуваємось колонок, зайвих для аналізу
df = df.drop(columns='Loan_ID')

# Кодування змінних
df = pd.get_dummies(df, drop_first=True).astype(int)
print(df.head().to_string(), '\n')
# Кореляція
num_df = df.select_dtypes(include='number')
corr = num_df.corr()['Loan_Status_Y'].sort_values(ascending=False).round(2)
print('Кореляція:\n', corr, '\n')
# Досить велика позитивна залежність між цільовою змінною Loan_Status_Y та Credit_History - 0.61

# Розробка моделі машинного навчання

# Розбиваємо дані на ф'ючі та ключову змінну
X = df.drop(columns = 'Loan_Status_Y')
y = df['Loan_Status_Y']

# Розбиваємо дані на тренуючі та тестові
X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=0, test_size=0.25, stratify=y)
param_grid = {'classifier__n_estimators': [50,100],
                'classifier__max_depth': [5,13],
                'classifier__max_features': [None,'sqrt'],
                'classifier__min_samples_split': [5,7],
                'classifier__min_samples_leaf': [3,5]
               } # Набір гіперпараметрів для пошуку найкращих

model = RandomForestClassifier(random_state=0)

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', model),
])

grid_search = GridSearchCV(pipeline, param_grid, cv=5) # Створюємо модель пошуку
grid_search.fit(X_train, y_train)
print("Найкращі параметри:", grid_search.best_params_, '\n')
best_model = grid_search.best_estimator_ # Робимо удосконалену модель
y_pred = best_model.predict(X_test)
# Оцінка моделі
print(f'Точність: {accuracy_score(y_test, y_pred) * 100:.2f}%')
print(f'Крос валідація: {cross_val_score(best_model, X_test, y_test, cv=5).mean() * 100:.2f}%')
print(f'F1 Score: {f1_score(X_test, y_test) * 100:.2f}%')

# Візуалізація даних

important_features = best_model.named_steps['classifier'].feature_importances_ # Важливі фічі

# Будуємо графік
plt.figure(figsize=(12,9))
sns.barplot(x = important_features, y = X.columns, hue =  X.columns, palette='Set2', legend=False)
plt.title('Важливість ознак')
plt.xlabel('Важливість')
plt.ylabel('Ознаки')
plt.tight_layout()
plt.show()

# Розподіл основних змінних
df.hist(figsize=(10,6))
plt.suptitle('Розподіл змінних')
plt.tight_layout()
plt.show()

# Зв’язок між декількома змінними та цільовою змінною
features = df[['Loan_Status_Y','Credit_History', 'LoanAmount', 'CoapplicantIncome', 'ApplicantIncome', 'Property_Area_Semiurban']]
corr = features.corr().round(2)
# Графік
plt.figure(figsize=(12,9))
sns.heatmap(data=corr, annot=True, cmap='coolwarm', square=True, linewidths=0.3)
plt.title('Кореляція між ознаками та цільовою змінною')
plt.xticks(rotation = 45)
plt.tight_layout()
plt.show()

# Зберігаємо модель у файл

joblib.dump(best_model, 'loan_model.pkl')
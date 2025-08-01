# Необхідні імпорти
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns

# Завантажуємо дані та робимо базовий аналіз і підготовку даних
df = pd.read_csv('loan_data.csv')
print(df.head().to_string())
print(f'Розподіл даних:\n{df.describe().round(2).to_string()}\n') # Розподіл даних
print(f'Відсутні значення:\n{df.isna().sum()}\n') # Відсутні значення
# Заповнюємо відсутні значення
for i in ['Gender', 'Dependents', 'Self_Employed']:
    df[i] = df[i].fillna(df[i].mode()[0])

df['Loan_Amount_Term'] = df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].median())
df['Credit_History'] = df['Credit_History'].fillna(-1) # Запонюємо пусті кредитні історії окремим числом
print(f'Відсутні значення:\n{df.isna().sum()}\n')
df['Loan_Status'] = df['Loan_Status'].map({'N': 0, 'Y': 1}) # Кодуємо цільову змінну

# Позбуваємось колонок, зайвих для аналізу
df = df.drop(columns='Loan_ID')

# Кореляція
# Створюємо числовий дф для перегляду кореляції

num_df = pd.get_dummies(df, drop_first=True).astype(int).select_dtypes(include='number')
corr_pearson = num_df.corr(method='pearson')['Loan_Status'].sort_values(ascending=False).round(2)
print(f'Кореляція з цільовою змінною:\n{corr_pearson}\n') # Кореляція Пірсона
corr_spearman = num_df.corr(method='spearman')['Loan_Status'].sort_values(ascending=False).round(2)
print(f'Кореляція з цільовою змінною:\n{corr_spearman}\n') # Кореляція Спірмана
corr_kendall = num_df.corr(method='kendall')['Loan_Status'].sort_values(ascending=False).round(2)
print(f'Кореляція з цільовою змінною:\n{corr_kendall}\n') # Кореляція Кендалла
# Є середня залежність між цільовою змінною Loan_Status та Credit_History за Спірманом - 0.4

# Розробка моделі машинного навчання

# Розбиваємо дані на ф'ючі та ключову змінну
X = df.drop(columns = 'Loan_Status')
y = df['Loan_Status']

# Розбиваємо дані на тренуючі та тестові
X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=0, test_size=0.25, stratify=y)

# Використовуємо модель "випадкового лісу"
model = RandomForestClassifier(random_state=0)

# Виділяємо окремо категоріальні та числові колонки
categorical_features = X.select_dtypes(include='object').columns.tolist()
numerical_features = X.select_dtypes(include='number').columns.tolist()

# Преобробка
preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), categorical_features),
    ('num', StandardScaler(), numerical_features)
])
# Конвеєр для зручної підготовки та створення моделі
pipeline = Pipeline([
    ('preprocessor',preprocessor),
    ('classifier', model),
])
param_grid = {
    'classifier__n_estimators': [100, 200, 300],
    'classifier__max_depth': [10, 20, None],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__min_samples_leaf': [1, 2, 4],
    'classifier__max_features': ['sqrt', 0.5, 'log2'],
    'classifier__bootstrap': [True, False],
    'classifier__class_weight': [None, 'balanced', 'balanced_subsample'],
    'classifier__ccp_alpha': [0.0, 0.005, 0.01]
} # Набір гіперпараметрів для пошуку найкращих
grid_search = GridSearchCV(pipeline, param_grid, cv=4) # Створюємо модель пошуку
grid_search.fit(X_train, y_train)
print("Найкращі параметри:", grid_search.best_params_, '\n')
best_model = grid_search.best_estimator_ # Робимо удосконалену модель
y_pred = best_model.predict(X_test)

# Оцінка моделі
print(f'Точність: {accuracy_score(y_test, y_pred) * 100:.2f}%')
print(f'Крос валідація: {cross_val_score(best_model, X_test, y_test, cv=5).mean() * 100:.2f}%')
print(f'Precision: {precision_score(y_test, y_pred) * 100:.2f}%')
print(f'Recall: {recall_score(y_test, y_pred) * 100:.2f}%')
print(f'F1 Score: {f1_score(y_test, y_pred) * 100:.2f}%')

# Візуалізація даних
important_features = best_model.named_steps['classifier'].feature_importances_ # Важливі фічі
feature_names = best_model.named_steps['preprocessor'].get_feature_names_out() # Імена ознак після трансформації

# Створюємо DataFrame з важливістю ознак
feat_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': important_features
}).sort_values(by='Importance', ascending=False)

# Будуємо графік
plt.figure(figsize=(12,9))
sns.barplot(data=feat_importance_df, x ='Importance', y='Feature', hue='Feature',  palette='Set2')
plt.title('Важливість ознак')
plt.xlabel('Важливість')
plt.ylabel('Ознаки')
plt.tight_layout()
plt.show()



# Розподіл змінних
df.hist(figsize=(10,6))
plt.suptitle('Розподіл змінних')
plt.tight_layout()
plt.show()

# Зв’язок між декількома змінними та цільовою змінною
features = df[['Loan_Status','Credit_History', 'Property_Area_Semiurban', 'CoapplicantIncome', 'ApplicantIncome', 'Married_Yes']]
# Використовуємо Спірмана, так як вона добре працює з нелійними залежностями
corr = features.corr(method='spearman').round(2)
# Графік
plt.figure(figsize=(12,9))
sns.heatmap(data=corr, annot=True, cmap='coolwarm', square=True, linewidths=0.3)
plt.title('Кореляція між ознаками та цільовою змінною')
plt.xticks(rotation = 45)
plt.tight_layout()
plt.show()

# Зберігаємо модель у файл
joblib.dump(best_model, 'loan_model.pkl')
print('Збережено!')
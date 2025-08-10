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

# Завантажуємо дані та робимо базовий аналіз
df = pd.read_csv('loan_data.csv')
print(df.head().to_string())
print(f'Розподіл даних:\n{df.describe().round(2).to_string()}\n')
print(f'Розмір набору даних:{df.shape}\n')
print(f'Типи даних по стовпцям:\n{df.dtypes}\n')

# Позбуваємося пустих значення Credit_History
print(f'Відсутні значення:\n{df.isna().sum()}\n')
df = df.dropna(subset=['Credit_History'])
# Інші заповнюємо модою та медіаною
for i in ['Gender', 'Dependents', 'Self_Employed']:
    df[i] = df[i].fillna(df[i].mode()[0])

df['Loan_Amount_Term'] = df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].median())
print(f'Відсутні значення:\n{df.isna().sum()}\n')

df['Loan_Status'] = df['Loan_Status'].map({'N': 0, 'Y': 1}) # Кодуємо цільову змінну

# Створюємо нові стовпці
df['First_Credit_Request'] = df['Credit_History'].map({0: True, 1: False}) # Цей буде показувати чи перший це кредит користувача
df['Total_Income'] = df['ApplicantIncome'] + df['CoapplicantIncome'] # Загальний дохід
df['Loan_Amount_by_Income'] = df['LoanAmount'] * 1000 / df['Total_Income'] # Відношенню суми позики до доходу

# Позбуваємось колонок, зайвих для аналізу
df = df.drop(columns=['Loan_ID'])

# Створюємо числовий дф для перегляду кореляції
num_df = pd.get_dummies(df, drop_first=True).astype(float).select_dtypes(include='number')
corr_pearson = num_df.corr(method='pearson')['Loan_Status'].sort_values(ascending=False).round(2)
print(f'Кореляція Пірсона з цільовою змінною:\n{corr_pearson}\n') # Кореляція Пірсона
corr_spearman = num_df.corr(method='spearman')['Loan_Status'].sort_values(ascending=False).round(2)
print(f'Кореляція Спірмана з цільовою змінною:\n{corr_spearman}\n') # Кореляція Спірмана
corr_kendall = num_df.corr(method='kendall')['Loan_Status'].sort_values(ascending=False).round(2)
print(f'Кореляція Кендалла з цільовою змінною:\n{corr_kendall}\n') # Кореляція Кендалла
# Є вище среднього негативна залежність між цільовою змінною Loan_Status та Credit_History - 0.62
# І також між цільовою змінною та First_Credit_Request протилежна - -0.62

# Розробка моделі машинного навчання
# Розбиваємо дані на ф'ючі та ключову змінну
X = df.drop(columns = 'Loan_Status')
y = df['Loan_Status']

# Розбиваємо дані на тренуючі та тестові
X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=0, test_size=0.25, stratify=y)

# Використовуємо модель "випадкового лісу"
model = RandomForestClassifier(random_state=0, bootstrap=True, class_weight='balanced')


# Виділяємо окремо категоріальні та числові колонки
categorical_features = X.select_dtypes(include='object').columns.tolist()
numerical_features = X.select_dtypes(include=['number','bool']).columns.tolist()

# Обробка даних
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
    'classifier__n_estimators': [300, 350],
    'classifier__max_depth': [3, 5],
    'classifier__min_samples_split': [4, 5],
    'classifier__min_samples_leaf': [1, 2],
    'classifier__max_features': ['sqrt', 0.5],
    'classifier__ccp_alpha': [0.025, 0.05]
} # Набір гіперпараметрів для пошуку найкращих

# Створюємо модель пошуку, оптимізуємо під F1 Score
grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=4, scoring='f1')
grid_search.fit(X_train, y_train)
print("Найкращі параметри:", grid_search.best_params_, '\n')
best_model = grid_search.best_estimator_ # Робимо удосконалену модель
y_pred = best_model.predict(X_test)

# Оцінка моделі
print(f'Точність: {accuracy_score(y_test, y_pred) * 100:.2f}%')
print(f'Кросс валідація: {cross_val_score(best_model, X_test, y_test, cv=5).mean() * 100:.2f}%')
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
df.hist(figsize=(12,8))
plt.suptitle('Розподіл змінних')
plt.tight_layout()
plt.show()

# Зв’язок між декількома змінними та цільовою змінною
features = df[['Loan_Status','Credit_History', 'First_Credit_Request', 'Total_Income', 'ApplicantIncome']]

# Використовуємо Спірмана, так як вона краще працює з нелійними залежностями
corr = features.corr(method='spearman').round(2)

# Графік
plt.figure(figsize=(12,9))
sns.heatmap(data=corr, annot=True, cmap='coolwarm', square=True, linewidths=0.3)
plt.title('Кореляція між ознаками та цільовою змінною')
plt.xticks(rotation = 45)
plt.tight_layout()
plt.show()

# Зберігаємо модель у файл
joblib.dump(best_model, 'LoanWebProject/loanapp/models/loan_model.pkl')
print('Збережено!')


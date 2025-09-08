import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc, roc_auc_score, confusion_matrix

data = pd.read_csv('teen_phone_addiction_dataset.csv')

# 1. Предобработка данных
print("Пропуски в данных:")
print(data.isnull().sum())

print("\nТипы данных:")
print(data.dtypes)

data['Addiction_Category'] = pd.cut(data['Addiction_Level'],
                                    bins=[0, 4, 7, 10],
                                    labels=['Low', 'Medium', 'High'])

data = data.drop('Addiction_Level', axis=1)

categorical_cols = ['Name', 'Gender', 'Location', 'School_Grade', 'Phone_Usage_Purpose']
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col].astype(str))
    label_encoders[col] = le

le_target = LabelEncoder()
data['Addiction_Category'] = le_target.fit_transform(data['Addiction_Category'])

# Разделение на признаки и целевую переменную
X = data.drop('Addiction_Category', axis=1)
y = data['Addiction_Category']

# Масштабирование числовых признаков
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. Анализ зависимостей
# Матрица корреляций
plt.figure(figsize=(12, 10))
sns.heatmap(data.corr(), cmap='coolwarm', annot=False)
plt.title('Матрица корреляций')
plt.show()

top_corr = data.corr()['Addiction_Category'].sort_values(ascending=False).head(10)
plt.figure(figsize=(10, 6))
sns.barplot(x=top_corr.values, y=top_corr.index)
plt.title('Наибольшие корреляции с Addiction_Category')
plt.show()

# 4. Обучение моделей
# Разделение данных
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# Логистическая регрессия с регуляризацией
lr = LogisticRegression(C=0.1, penalty='l2', random_state=42)
lr.fit(X_train, y_train)

# Случайный лес с настройкой гиперпараметров
rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf.fit(X_train, y_train)

# Градиентный бустинг с регуляризацией
xgb = XGBClassifier(learning_rate=0.1, n_estimators=100, max_depth=5, random_state=42)
xgb.fit(X_train, y_train)

# 5. Оценка моделей
models = {'Logistic Regression': lr, 'Random Forest': rf, 'XGBoost': xgb}
results = {}

plt.figure(figsize=(10, 8))
for name, model in models.items():
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=le_target.classes_, zero_division=0)
    results[name] = {
        'accuracy': accuracy,
        'report': report
    }
    print(f'{name} Accuracy: {accuracy:.4f}')
    print(report)

    # ROC-кривая
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)
        for i in range(len(le_target.classes_)):
            fpr, tpr, _ = roc_curve((y_test == i).astype(int), y_prob[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'{name} (Class {le_target.classes_[i]}, AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC-кривые (многоклассовая классификация)')
plt.legend(loc='lower right')
plt.show()

# Вывод лучшей модели
best_model_name = max(results, key=lambda x: results[x]['accuracy'])
print(f'Лучшая модель: {best_model_name} с точностью {results[best_model_name]["accuracy"]:.4f}')
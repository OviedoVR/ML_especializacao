# **Cheat Sheet de Machine Learning com Scikit-Learn e Pandas**

## 1. `train_test_split` (do `sklearn`)

```python
from sklearn.model_selection import train_test_split

# Dados fictícios
X = features
y = target

# Dividindo os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

## 2. Encoding com `pd.get_dummies()` e `OneHotEncoder`

### `pd.get_dummies()`

```python
import pandas as pd

# Dados fictícios
df = pd.DataFrame({'pay_method': ['credit', 'debit', 'pay_online']})

# Codificação
df_encoded = pd.get_dummies(df, columns=['pay_method'])
```

### `OneHotEncoder`

```python
from sklearn.preprocessing import OneHotEncoder
import numpy as np

# Dados fictícios
encoder = OneHotEncoder(sparse=False)
X = np.array(['credit', 'debit', 'pay_online']).reshape(-1, 1)

# Codificação
X_encoded = encoder.fit_transform(X)
```

## 3. Normalização com `StandardScaler`

```python
from sklearn.preprocessing import StandardScaler

# Dados fictícios
scaler = StandardScaler()
X = ...

# Normalização
X_scaled = scaler.fit_transform(X)
```

## 4. Normalização com `MinMaxScaler`

```python
from sklearn.preprocessing import MinMaxScaler

# Dados fictícios
scaler = MinMaxScaler()
X = ...

# Normalização
X_scaled = scaler.fit_transform(X)
```

## 5. Métricas de Regressão com `sklearn`

```python
from sklearn.metrics import mean_squared_error, r2_score

# Dados fictícios
y_true = ...
y_pred = ...

# Métricas
mse = mean_squared_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)
```

## 6. Métricas de Classificação com `sklearn`

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# Dados fictícios
y_true = ...
y_pred = ...

# Métricas
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='binary')
recall = recall_score(y_true, y_pred, average='binary')
f1 = f1_score(y_true, y_pred, average='binary')
conf_matrix = confusion_matrix(y_true, y_pred)
report = classification_report(y_true, y_pred)
```

## 7. K-means

```python
from sklearn.cluster import KMeans

# Dados fictícios
X = ...

# Treinamento do modelo
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)

# Predições
labels = kmeans.predict(X)
```

## 8. Simple Linear Regression (`sklearn`)

```python
from sklearn.linear_model import LinearRegression

# Dados fictícios
X, y = ..., ...

# Treinamento do modelo
model = LinearRegression()
model.fit(X, y)

# Predições
y_pred = model.predict(X)
```

## 9. Polynomial Regression (`sklearn`)

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Dados fictícios
X, y = ..., ...

# Transformação polinomial
poly = PolynomialFeatures(degree=3)
X_poly = poly.fit_transform(X)

# Treinamento do modelo
model = LinearRegression()
model.fit(X_poly, y)

# Predições
y_pred = model.predict(X_poly)
```

## 10. Logistic Regression (`sklearn`)

```python
from sklearn.linear_model import LogisticRegression

# Dados fictícios
X, y = ..., ...

# Treinamento do modelo
model = LogisticRegression()
model.fit(X, y)

# Predições
y_pred = model.predict(X)
```

## 11. Decision Tree Classifier (`sklearn`)

```python
from sklearn.tree import DecisionTreeClassifier

# Dados fictícios
X, y = ..., ...

# Treinamento do modelo
model = DecisionTreeClassifier()
model.fit(X, y)

# Predições
y_pred = model.predict(X)
```

## 12. Decision Tree Regressor (`sklearn`)

```python
from sklearn.tree import DecisionTreeRegressor

# Dados fictícios
X, y = ..., ...

# Treinamento do modelo
model = DecisionTreeRegressor()
model.fit(X, y)

# Predições
y_pred = model.predict(X)
```

## 13. Random Forest

```python
from sklearn.ensemble import RandomForestClassifier

# Dados fictícios
X, y = ..., ...

# Treinamento do modelo
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Predições
y_pred = model.predict(X)
```

## 14. Naive Bayes

### Gaussian Naive Bayes

```python
from sklearn.naive_bayes import GaussianNB

# Dados fictícios
X, y = ..., ...

# Treinamento do modelo
model = GaussianNB()
model.fit(X, y)

# Predições
y_pred = model.predict(X)
```

## 15. K-Nearest Neighbors (KNN)

```python
from sklearn.neighbors import KNeighborsClassifier

# Dados fictícios
X, y = ..., ...

# Treinamento do modelo
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X, y)

# Predições
y_pred = model.predict(X)
```

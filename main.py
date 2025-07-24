import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, precision_recall_curve
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, learning_curve
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
import joblib



# Chargement des données de transaction
data = pd.read_csv('transactions.csv')

# Suppression des colonnes non pertinentes ou redondantes :
# - 'nameOrig' et 'nameDest' : identifiants sans valeur prédictive utile
# - 'step' : équivalent temporel peu informatif selon nos tests
data = data.drop(['nameOrig', 'nameDest', 'step'], axis=1)

# définition variables X et y
X = data.drop('isFraud', axis=1)
y = data['isFraud']

print(X.columns)
input()
# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# Conversion explicite en DataFrame pour conserver les noms de colonnes après le split
X_train = pd.DataFrame(X_train, columns=X.columns)
X_test = pd.DataFrame(X_test, columns=X.columns)
y_train = pd.Series(y_train)
y_test = pd.Series(y_test)

# Fonction pour afficher les principales métriques d’évaluation du modèle
def evaluation(y_true, y_pred):
    print('accuracy :', accuracy_score(y_true, y_pred))
    print('f1:', f1_score(y_true, y_pred))
    print('precision :', precision_score(y_true, y_pred))
    print('recall :', recall_score(y_true, y_pred))


# Séparation des variables numériques et catégorielles pour appliquer un prétraitement spécifique
num_columns = X.select_dtypes(include=[np.float64, np.int64]).columns
cat_columns = X.select_dtypes(include=object).columns


# définition pipeline numérique
num_pipeline = Pipeline([
    ('scaler', StandardScaler())
])

# définition pipeline catégorielle
cat_pipeline = Pipeline([
    ('encoder_OH', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
])


# définition pipeline pour le preprocessing
preprocessor = ColumnTransformer([
    ('num', num_pipeline, num_columns),
    ('cat', cat_pipeline, cat_columns)

])

# définition pipeline finale (preprocessor + model)
model = Pipeline([
    ('preprocessor', preprocessor),
    ('model', RandomForestClassifier(random_state=0)) 
])


# Définition d’une validation croisée stratifiée pour équilibrer les classes
cv_SKF = StratifiedKFold(n_splits=4, random_state=0, shuffle=True)

# Recherche d’hyperparamètres du modèle avec GridSearchCV
param_grid = {
    'model__n_estimators' : [100 , 200],
}
grid = GridSearchCV(model, param_grid=param_grid, cv=cv_SKF, n_jobs=-1, verbose=2, scoring='f1')

# entraînement modèle
grid.fit(X_train, y_train)
model = grid.best_estimator_

# affichage des meilleurs hyper-paramètres
print(grid.best_params_)


# Fonction pour convertir les probabilités en classes selon un seuil personnalisé
def predict(threshold, y_pred):
    return (y_pred > threshold).astype(int)

# génération des prédictions
y_pred = predict(0.40, model.predict_proba(X_test)[:, 1])

# évaluation de notre modèle
evaluation(y_test, y_pred)

# enregistrement du modèle avec joblib
joblib.dump(model, 'model_fraude_bancaire.pkl')

# learning_curve
N, train_score, val_score = learning_curve(model, X_train, y_train, train_sizes=np.arange(0.1, 1, 0.1), cv=cv_SKF, n_jobs=-1, scoring='f1')
plt.plot(N, train_score.mean(axis=1), label='train')
plt.plot(N, val_score.mean(axis=1), label='test')
plt.legend()
plt.show()


# Idées testées mais non approuvées n'apparaîssant pas dans le code : 
# 1) PCA (baisse significative du score)
# 2) suppression des outliers avec IsolationForest (baisse significative du score)
# 3) feature engineering : utilisation de PolynomialFeatures (baisse significative du score)





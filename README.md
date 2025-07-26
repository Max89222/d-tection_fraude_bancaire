# Création d'un modèle de classification binaire capable de détecter la présence de fraudes bancaires

**Auteur : Max89222**

---

Ce projet vise à prédire la présence de fraudes bancaires à partir de données sur la nature de la transaction

## 1) Structure et utilité des fichiers

- `transactions.csv` : fichier dans lequel chaque ligne représente une transaction et chaque colonne une caractéristique de cette transaction
- `main.py` : fichier python contenant le code source de notre modèle
- `model_fraude_bancaire.pkl`: fichier permettant d'enregistrer le modèle déjà entraîné dans le but d'effectuer des prédictions sans avoir à l'entraîner à nouveau
- `app.py` : interface graphique permettant d'effectuer des prédictions en temps réel

## 2) Dataset

taille du dataset : (199999, 10)

 📊 Description des variables du dataset sur le diabète

| Feature           | Description                                                                 |
|-------------------|-----------------------------------------------------------------------------|
| `type`            | Type de transaction (ex. : CASH_OUT, TRANSFER, DEBIT, etc.)                |
| `amount`          | Montant de la transaction                                                   |
| `oldbalanceOrg`   | Solde du compte émetteur **avant** la transaction                          |
| `newbalanceOrig`  | Solde du compte émetteur **après** la transaction                          |
| `oldbalanceDest`  | Solde du compte destinataire **avant** la transaction                      |
| `newbalanceDest`  | Solde du compte destinataire **après** la transaction                      |
| `isFraud`         | Indique si la transaction est frauduleuse (1 = fraude, 0 = non fraude)     |



## 3) Technologies utilisées

- `pandas` : manipulation et nettoyage des données  
- `scikit-learn` : création et évaluation des modèles de machine learning  
- `matplotlib` : visualisation des données  
- `numpy` : opérations numériques  
- `joblib` : sauvegarde et chargement des modèles

Le modèle final utilisé est un **RandomForestClassifier**, qui s’est avéré le plus performant dans ce problème de classification binaire.

## 4) Résultats et métriques

voici les scores obtenus sur le test set : 

- accuracy : 0.999675
- f1: 0.8686868686868687
- precision : 0.9772727272727273
- recall : 0.7818181818181819
- 
## 5) Installation

1. Installer Git (si ce n’est pas déjà fait) :
   
`brew install git`

2. Cloner le dépôt :

`git clone <clé_ssh>`
`cd <nom_du_dossier>`

3. Installer les dépendances :

`pip3 install pandas scikit-learn matplotlib numpy joblib`

4. Entraîner le modèle :

-Pour entraîner le modèle vous même (pas utile à part si vous souhaitez voir le processus d'entraînement du modèle) : 
Ouvrir `main.py` (ainsi que les autres fichiers) dans un éditeur de code et l'exécuter

-Pour effectuer des prédictions : 
Si vous souhaitez effectuer des prédictions, il vous suffit simplement d'exécuter le fichier `app.py` . Une interface graphique va alors s'ouvrir, vous permettant de prédire si une transaction est une fraude ou non à partir d'informations que vous fournirez


## 6) Idées d'amélioration et contributions
N'hésitez pas à contribuer à ce projet ou à proposer des idées d'amélioration !

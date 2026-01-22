# Prédiction de la note globale des joueurs de football par Deep Learning

## 1. Contexte et objectif du projet

Ce projet s’inscrit dans un cadre d’apprentissage et de mise en pratique du **Deep Learning appliqué au sport**, plus précisément au football.

L’objectif principal est de **prédire la note globale (`overall_rating`) d’un joueur de football** à partir de ses caractéristiques techniques, physiques et de jeu, à l’aide du framework **PyTorch**.

Ce type de problématique est inspiré de cas réels rencontrés dans :

* l’analyse de performance sportive,
* le scouting,
* l’évaluation et la comparaison de joueurs.

---

## 2. Description du dataset

Le dataset utilisé est un dataset de type **FIFA players**, contenant :

* **17 954 joueurs**
* **51 colonnes** décrivant les attributs des joueurs

### Variable cible

* `overall_rating` : note globale du joueur (score continu)

### Types de variables

* **Numériques** : statistiques techniques (passes, tirs, défense…), caractéristiques physiques
* **Catégorielles** : pied préféré, position principale

Certaines variables ont été volontairement exclues afin d’éviter toute **fuite de données**.

---

## 3. Choix techniques et justification

### 3.1 Choix du framework

* **PyTorch** a été choisi pour sa flexibilité, sa clarté et sa proximité avec les concepts fondamentaux du Deep Learning.

### 3.2 Choix du modèle

* Problème formulé comme une **régression**
* Modèle : **MLP (Multi-Layer Perceptron)**

Architecture retenue :

* Couches fully connected
* Fonctions d’activation ReLU
* Dropout pour limiter l’overfitting

Ce choix est adapté aux **données tabulaires** et permet une bonne interprétabilité.

### 3.3 Choix des features

Conservées :

* Attributs techniques (`crossing` → `sliding_tackle`)
* Variables physiques (`age`, `height_cm`, `weight_kgs`)
* Méta-attributs (`weak_foot`, `skill_moves`, `international_reputation`)

Exclues :

* Identifiants (`name`, `full_name`)
* Variables économiques (`value_euro`, `wage_euro`) pour éviter la fuite de données
* Variables trop incomplètes (sélection nationale)

---

## 4. Pipeline de traitement des données

1. Chargement du CSV
2. Nettoyage des données
3. Gestion des valeurs manquantes (médiane / valeur par défaut)
4. Encodage one-hot des variables catégorielles
5. Normalisation des variables numériques
6. Split : train / validation / test

Ce pipeline est implémenté dans le fichier `dataset.py`.

---

## 5. Entraînement du modèle

### Paramètres principaux

* Loss : `SmoothL1Loss`
* Optimiseur : `AdamW`
* Early stopping pour éviter l’overfitting

### Métrique d’évaluation

* **MAE (Mean Absolute Error)**

---

## 6. Résultats obtenus

### 6.1 Métriques quantitatives

Les performances du modèle ont été évaluées sur un jeu de test totalement indépendant.

* **MAE (Mean Absolute Error) test ≈ 0.93**
* **MAE validation minimale ≈ 0.91**

Ces valeurs signifient que le modèle se trompe en moyenne de **moins d’un point** sur la note globale d’un joueur, ce qui constitue une très bonne performance pour un problème de régression sur données tabulaires.

La proximité entre la MAE de validation et la MAE de test indique une **bonne capacité de généralisation**, sans surapprentissage significatif.

---

### 6.2 Analyse visuelle des résultats

Plusieurs graphiques ont été générés afin de mieux interpréter les performances du modèle.

#### a) Vrai vs Prédit

Ce graphique compare la note réelle (`overall_rating`) et la note prédite par le modèle. La proximité des points avec la diagonale indique une forte corrélation entre les valeurs réelles et prédites.

<img width="767" height="574" alt="image" src="https://github.com/user-attachments/assets/12ace383-4b89-4dfa-8bd3-fb715c34c562" />


#### b) Distribution des erreurs absolues

Cet histogramme montre la répartition des erreurs absolues. La majorité des erreurs est concentrée sous le seuil de **1 point**, confirmant la bonne précision du modèle.

<img width="811" height="610" alt="image" src="https://github.com/user-attachments/assets/c41588e5-e6bd-4c6f-9792-5fd70c402c08" />


#### c) Courbe cumulative des erreurs

Cette courbe indique la proportion de prédictions dont l’erreur absolue est inférieure à un seuil donné. Plus de **90 %** des prédictions présentent une erreur inférieure à **1.5**.

<img width="809" height="606" alt="image" src="https://github.com/user-attachments/assets/a112f5d3-47ae-4965-a7fc-4f02610ea1dd" />


Ces visualisations renforcent l’analyse quantitative et rendent les résultats facilement interprétables dans un rapport ou une soutenance.

---

### 6.3 Analyse des erreurs

Une analyse quantitative détaillée des erreurs a été réalisée à partir du fichier `test_predictions.csv`, généré lors de l’évaluation finale du modèle.

#### a) Indicateurs globaux d’erreur

* **MAE (Mean Absolute Error)** : **0.93**
* **Erreur absolue médiane** : **0.60**
* **90e percentile de l’erreur absolue** : **1.86**
* **95e percentile de l’erreur absolue** : **3.15**

Ces statistiques montrent que :

* plus de 50 % des prédictions ont une erreur inférieure à **0.6** point,
* 90 % des prédictions ont une erreur inférieure à **1.9** points,
* seules quelques prédictions rares dépassent **3** points d’erreur.

---

#### b) Cas de plus fortes erreurs

L’analyse des **pires erreurs** (Top 20) met en évidence des écarts compris entre **6.3 et 10 points** entre la note réelle et la note prédite.

Exemples représentatifs :

| Note réelle | Note prédite | Erreur absolue |
| ----------: | -----------: | -------------: |
|          82 |         72.0 |           9.99 |
|          54 |         63.7 |           9.71 |
|          62 |         71.6 |           9.58 |
|          70 |         61.0 |           9.02 |
|          76 |         68.8 |           7.24 |

Ces erreurs importantes concernent majoritairement :

* des joueurs aux profils **atypiques ou hybrides**,
* des joueurs dont les statistiques techniques ne reflètent pas totalement la note globale,
* des profils rares ou sous-représentés dans le dataset.

---

#### c) Interprétation

Malgré ces quelques cas extrêmes, leur faible proportion n’affecte pas significativement la performance globale du modèle. Cette analyse met en évidence la robustesse du modèle sur la majorité des joueurs, tout en soulignant des pistes d’amélioration futures (segmentation par poste, enrichissement des features).

---

### 6.4 Interprétation globale

Les résultats obtenus montrent que le modèle est :

* **précis** (erreur moyenne faible),
* **stable** (faible écart validation / test),
* **robuste** face à la majorité des profils de joueurs.

Il constitue une base solide pour des extensions plus avancées, telles que l’analyse par poste ou l’intégration d’embeddings.

---

## 7. Structure du projet

```text
football-rating-ml/
 ├─ data/
 │   └─ raw/
 │       └─ players.csv
 ├─ notebooks/
 │   └─ 01_data_exploration.ipynb
 ├─ src/
 │   ├─ dataset.py
 │   ├─ model.py
 │   ├─ train.py
 │   └─ sanity_check.py
 ├─ outputs/
 │   └─ models/
 │       └─ best_model.pt
 ├─ requirements.txt
 └─ README.md
```

---

### 8. ### Prédictions interprétables (lecture humaine)

Afin de rendre les résultats du modèle compréhensibles par des non-spécialistes, les prédictions sont présentées sous forme de tableau, accompagnées d’une interprétation qualitative.

| Note réelle | Note prédite | Erreur absolue | Interprétation |
|------------:|-------------:|---------------:|----------------|
| 70.0 | 61.0 | 9.02 | Prédiction imprécise |
| 72.0 | 70.1 | 1.87 | Prédiction acceptable |
| 72.0 | 72.5 | 0.48 | Prédiction très précise |
| 56.0 | 56.4 | 0.38 | Prédiction très précise |
| 71.0 | 68.0 | 3.02 | Prédiction imprécise |

**Lecture du tableau :**
- Une erreur inférieure à **0.5** correspond à une prédiction **très précise**.
- Une erreur comprise entre **1 et 2** correspond à une prédiction **acceptable**.
- Les erreurs supérieures à **3** concernent des **cas atypiques ou rares**.

Ce format permet de traduire les sorties du modèle en **résultats clairs et exploitables**, compréhensibles même par des utilisateurs non techniques (analystes sportifs, recruteurs, décideurs).





## 9. Améliorations possibles

* Utilisation d’**embeddings** pour les variables catégorielles
* Ajout de nouvelles features agrégées (attaque / défense / physique)
* Analyse d’erreurs détaillée par poste
* Prédiction du **potentiel (`potential`)**
* Classification des joueurs (élite vs non-élite)

---

## 10. Conclusion

Ce projet démontre la mise en place complète d’un **pipeline Deep Learning professionnel**, depuis la préparation des données jusqu’à l’évaluation finale du modèle.

Il constitue une base solide pour :

* un projet académique,
* un portfolio technique,
* ou des travaux futurs en IA appliquée au sport.

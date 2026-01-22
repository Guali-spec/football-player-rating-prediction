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

### Meilleure performance validation

* **MAE validation ≈ 0.91**

### Performance finale sur le jeu de test

* **MAE test ≈ 0.93**

### Interprétation

La prédiction de la note globale du joueur présente une erreur moyenne inférieure à **1 point**, ce qui est considéré comme **une très bonne performance** pour ce type de données.

La proximité entre les scores validation et test montre que le modèle **généralise correctement**.

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

## 8. Améliorations possibles

* Utilisation d’**embeddings** pour les variables catégorielles
* Ajout de nouvelles features agrégées (attaque / défense / physique)
* Analyse d’erreurs détaillée par poste
* Prédiction du **potentiel (`potential`)**
* Classification des joueurs (élite vs non-élite)

---

## 9. Conclusion

Ce projet démontre la mise en place complète d’un **pipeline Deep Learning professionnel**, depuis la préparation des données jusqu’à l’évaluation finale du modèle.

Il constitue une base solide pour :

* un projet académique,
* un portfolio technique,
* ou des travaux futurs en IA appliquée au sport.

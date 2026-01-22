# Checklist de reprise de l’environnement de travail (après redémarrage)

Ce document sert de **guide de reprise systématique** après chaque redémarrage de l’ordinateur, afin d’éviter les erreurs classiques (venv non activé, mauvais chemin, mauvais interpréteur Python, etc.).

---

## 1. Ouvrir un terminal et se placer à la racine du projet

Ouvre **PowerShell**, puis navigue vers le dossier racine du projet :

```powershell
cd D:\L3\DeepLearning\Deep_Project\football-rating-ml
```

⚠️ Toujours travailler **depuis la racine du projet**, jamais depuis `src/` ou `notebooks/`.

---

## 2. Activer l’environnement virtuel (OBLIGATOIRE)

Après chaque redémarrage, l’environnement virtuel est désactivé par défaut.

```powershell
.\.venv\Scripts\Activate.ps1
```

✅ Vérification : le terminal doit afficher `(.venv)` au début de la ligne.

---

## 3. Vérifier que l’environnement est sain

Exécuter les commandes suivantes :

```powershell
python --version
pip --version
```

Puis :

```powershell
python -c "import torch, pandas, sklearn; print('OK')"
```

✅ Si `OK` s’affiche, l’environnement Python est prêt.

---

## 4. Ouvrir VS Code depuis le bon dossier

Toujours depuis la racine du projet :

```powershell
code .
```

Dans VS Code :

- Vérifier que l’interpréteur Python sélectionné est **.venv**
- Ouvrir le terminal intégré et vérifier que `(.venv)` est affiché

---

## 5. Vérifier la structure minimale du projet

La structure suivante doit être présente :

```text
football-rating-ml/
 ├─ .venv/
 ├─ data/
 │   └─ raw/
 │       └─ players.csv
 ├─ notebooks/
 ├─ src/
 │   └─ sanity_check.py
 ├─ requirements.txt
 └─ .gitignore
```

⚠️ Si `players.csv` n’est pas dans `data/raw/`, ne pas continuer avant correction.

---

## 6. Lancer le test de reprise (sanity check)

Toujours depuis la racine du projet :

```powershell
python src\sanity_check.py
```

Résultat attendu :

- Version de PyTorch affichée
- CUDA `False` (normal si CPU)
- CSV chargé sans erreur
- Liste des colonnes affichée

---

## 7. (Optionnel) Lancer Jupyter Notebook

Si une phase d’exploration des données est prévue :

```powershell
jupyter notebook
```

Ouvrir ensuite :

```text
notebooks/01_data_exploration.ipynb
```

---

## 8. Règle d’or à ne jamais oublier

> Après chaque redémarrage :
>
> 1. Se placer à la racine du projet
> 2. Activer le `.venv`
> 3. Vérifier l’environnement
> 4. Lancer le sanity check
> 5. Seulement ensuite coder

---

##

Ce document est à conserver et à suivre **systématiquement** pour travailler comme un ingénieur IA professionnel.


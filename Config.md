CHECKLIST DE REPRISE (APRÈS REDÉMARRAGE)
🔹 Étape 1 — Ouvrir le projet correctement

Ouvre PowerShell

Va dans le dossier du projet :

cd D:\L3\DeepLearning\Deep_Project\football-rating-ml


⚠️ Important : toujours te placer à la racine du projet

🔹 Étape 2 — Réactiver l’environnement virtuel

Après un redémarrage, le venv est toujours désactivé.

.\.venv\Scripts\Activate.ps1


✅ Tu dois voir (venv) ou (.venv) au début du terminal.

👉 Si tu oublies cette étape, tout le reste casse.

🔹 Étape 3 — Vérifier que l’environnement est sain (30 secondes)

Lance ces commandes une par une :

python --version
pip --version


Puis :

python -c "import torch, pandas, sklearn; print('OK')"


✅ Si tu vois OK, ton environnement est prêt.

🔹 Étape 4 — Ouvrir VS Code depuis le BON dossier

Toujours depuis la racine :

code .


Dans VS Code :

vérifie que l’interpréteur Python est bien .venv

ouvre le terminal intégré → il doit afficher (.venv)

🔹 Étape 5 — Vérifier la présence des fichiers clés

Dans l’explorateur VS Code, tu dois voir :

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


⚠️ Si players.csv n’est pas là → on s’arrête et on corrige.

🔹 Étape 6 — Lancer le test de reprise (obligatoire)

Toujours à la racine :

python src\sanity_check.py


✅ Résultat attendu :

version PyTorch

CUDA False (normal)

CSV chargé sans erreur

affichage des colonnes

🔹 Étape 7 — (Optionnel) Lancer Jupyter si on continue l’exploration

Si on repart sur l’analyse des données :

jupyter notebook


Puis ouvre :

notebooks/01_data_exploration.ipynb

🧠 RÈGLE D’OR À RETENIR (très importante)

Après chaque redémarrage :

cd vers le projet

activer le .venv

vérifier l’interpréteur

seulement ensuite coder

C’est une habitude professionnelle.
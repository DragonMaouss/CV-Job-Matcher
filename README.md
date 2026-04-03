# CV-Job-Matcher

Projet de matching intelligent de CV et offres d'emploi en python.

## Getting Started

### Prérequis

- Python 3.x
- Pip
- Git

### Installation

- Cloner le projet 
  
```bash
git clone https://github.com/DragonMaouss/CV-Job-Matcher.git
cd CV-Job-Matcher
```

### Créaction environnement virtuel

```bash
python3 -m venv .venv
```
- Activation :
  
```bash
. .venv/bin/activate
```

- Désactivation :
  
```bash
deactivate
```

### Installer les dépendances 
  
```bash
pip install -r requirements.txt
```

### Lancer le projet

```bash
python main.py
```

## Description

Cette première version permet de comparer automatiquement des CV et des offres d’emploi à partir de leur contenu textuel.

Le pipeline actuel :
- récupère un dataset de recrutement
- extrait les colonnes CV et descriptions de poste
- nettoie les textes
- vectorise les contenus avec TF-IDF
- calcule une similarité cosine entre CV et offres
- retourne le meilleur match pour chaque CV
  

## Structure du projet 

```bash
.
├── data/
│   ├── resumes.csv
│   └── jobs.csv
│   └── matching_results.csv
├── src/
│   ├── preprocess.py
│   ├── matcher.py
│   └── main.py
└── README.md
```

# Documentation Complète du Notebook : Détection de Défauts sur Drones

## 1. Introduction et Objectifs

Ce notebook a pour but d'analyser le dataset **DronePropA** afin de développer un système de **maintenance prédictive** pour les drones. L'objectif est de détecter, identifier et quantifier les défauts sur les hélices à partir des données des capteurs de vol.

Les **3 objectifs principaux** sont :
1.  **Détection de Défaut** : Classifier un vol comme "Sain" ou "Défectueux".
2.  **Identification du Type de Défaut** : Déterminer la nature du défaut parmi 4 classes (Sain, Coupure de bord, Fissure, Coupure de surface).
3.  **Évaluation de la Sévérité** : Estimer le niveau de gravité du défaut sur une échelle de 0 (sain) à 3 (fort).

## 2. Configuration de l'Environnement

La première cellule du notebook importe les bibliothèques Python essentielles pour l'analyse :
-   `pandas` & `numpy` : Pour la manipulation des données (tableaux, matrices).
-   `scipy.io` : Pour charger les fichiers de données au format `.mat` (format MATLAB).
-   `os` & `pathlib` : Pour la gestion des fichiers et des chemins dans le système.
-   `matplotlib` & `seaborn` : Pour la création de graphiques et la visualisation des données.
-   `warnings` : Pour masquer les avertissements non critiques et garder une sortie propre.

## 3. Exploration et Chargement des Données

### 3.1. Structure des Fichiers
Le dataset est composé de **130 fichiers `.mat`**. Chaque nom de fichier contient des informations précieuses sur les conditions de vol :
-   `F{type}`: Type de défaut (0 pour sain).
-   `SV{gravité}`: Sévérité du défaut.
-   `SP{vitesse}`: Vitesse de vol.
-   `t{trajectoire}`: Type de trajectoire effectuée.

### 3.2. Contenu d'un Fichier `.mat`
Chaque fichier `.mat` contient 3 matrices principales qui représentent des séries temporelles de données de capteurs enregistrées à **1000 Hz** :
1.  `commander_data`: Données de trajectoire (positions de référence et mesurées).
2.  `QDrone_data`: Données des capteurs embarqués (IMU, gyroscopes, accéléromètres, moteurs). C'est la source d'information la plus riche.
3.  `stabilizer_data`: Mode de vol.

### 3.3. Conversion en DataFrame
Une fonction `mat_to_dataframe` est définie pour :
1.  Charger un fichier `.mat`.
2.  Extraire les séries temporelles de chaque capteur.
3.  Les organiser dans un **DataFrame `pandas`**, où chaque colonne représente un capteur et chaque ligne un instant `t`.
4.  Ajouter les métadonnées (type de défaut, sévérité, etc.) extraites du nom du fichier.

Cette conversion permet de passer d'un format MATLAB à un format tabulaire standard, plus facile à manipuler en Python.

## 4. Prétraitement et Feature Engineering

C'est l'étape la plus cruciale du notebook. Au lieu d'utiliser les millions de points de données brutes, nous extrayons des **caractéristiques (features) statistiques** qui résument l'information de chaque vol.

### 4.1. Sélection des Capteurs Pertinents
Une analyse approfondie est menée pour décider quelles colonnes (capteurs) garder.
-   **Capteurs Gardés (les plus importants)** :
    -   **Gyroscopes (12 colonnes)** : Mesurent les vitesses de rotation. Ils sont **critiques** car un défaut d'hélice provoque des vibrations et des oscillations anormales.
    -   **Accéléromètres (12 colonnes)** : Mesurent les accélérations. Ils sont **essentiels** pour détecter les forces asymétriques causées par une poussée inégale.
-   **Capteurs Éliminés** :
    -   **Données de référence et de position Optitrack** : Ce sont des consignes externes ou des mesures de position absolue, qui ne donnent pas d'information directe sur l'état mécanique du drone.
    -   **Commandes moteurs** : Elles sont une *conséquence* du défaut (le contrôleur qui essaie de compenser) et sont donc redondantes avec les mesures des gyroscopes/accéléromètres qui sont la *cause*.
    -   **Données constantes ou peu variables** (`Flight_Mode`, `Battery_Level` sur des vols courts).

### 4.2. Extraction de Features Agrégées
Pour chaque fichier de vol (et donc pour chaque capteur sélectionné), trois types de caractéristiques sont calculées :
1.  **Features Temporelles (12)** : `mean`, `std`, `min`, `max`, `rms`, `skewness`, `kurtosis`, etc. Elles décrivent la distribution statistique du signal dans le temps.
2.  **Features Fréquentielles (7)** : Basées sur la **Transformée de Fourier Rapide (FFT)**, elles analysent le contenu en fréquence du signal. Des features comme l'énergie dans certaines bandes de fréquences (`fft_energy_50_100hz`) ou la fréquence de pic (`fft_peak_freq`) sont très efficaces pour identifier des signatures vibratoires anormales.
3.  **Features Dynamiques (3)** : `zero_crossings` (nombre de fois où le signal passe par zéro), `autocorrélation`. Elles décrivent le comportement dynamique et la périodicité du signal.

Au final, pour chacun des **130 vols**, on extrait **528 features** (24 capteurs × 22 features/capteur). Le dataset passe de plus de 10 millions de lignes à un tableau propre de **130 lignes × 535 colonnes** (features + métadonnées).

## 5. Analyse Exploratoire des Données (EDA)

Avant de passer à la modélisation, le notebook visualise les données agrégées pour confirmer leur pertinence :
-   **Distribution des classes** : On vérifie que le nombre d'échantillons pour chaque type de défaut et chaque niveau de sévérité est équilibré.
-   **Boxplots** : Des graphiques comparent la distribution de features clés (comme le `RMS` des gyroscopes) entre les vols sains et défectueux. Ces graphiques montrent clairement des différences significatives, confirmant que les features choisies sont discriminantes.
-   **Matrice de Corrélation** : Elle montre comment les features sont liées entre elles. On observe par exemple une forte corrélation entre les capteurs redondants (IMU1 et IMU2), ce qui est attendu et confirme la qualité des données.

## 6. Modélisation Machine Learning

Le notebook aborde les 3 tâches de classification définies au début.

### 6.1. Préparation
-   **Séparation des données** : Le dataset de 130 échantillons est divisé en un ensemble d'**entraînement (80%)** et un ensemble de **test (20%)**.
-   **Normalisation** : Les features sont normalisées (`StandardScaler`) pour que les modèles ne soient pas biaisés par des échelles de valeurs différentes.

### 6.2. Entraînement et Évaluation
Deux modèles de classification populaires sont entraînés et comparés pour chaque tâche :
1.  **Random Forest Classifier** : Un modèle d'ensemble basé sur des arbres de décision, robuste et performant.
2.  **XGBoost Classifier** : Un autre modèle basé sur des arbres (gradient boosting), souvent considéré comme l'un des plus performants pour les données tabulaires.

Pour chaque tâche, les performances sont évaluées à l'aide de :
-   **Accuracy (Précision)** : Le pourcentage de prédictions correctes.
-   **Rapport de Classification** : Détaille la `précision`, le `rappel` et le `f1-score` pour chaque classe.
-   **Matrice de Confusion** : Une table qui montre en détail les erreurs du modèle (par exemple, quand un vol "Sain" est prédit comme "Défectueux").

### 6.3. Résultats Obtenus
-   **Tâche 1 (Détection)** : Les modèles atteignent une **excellente précision (autour de 77-81%)**, montrant qu'il est tout à fait possible de distinguer un drone sain d'un drone défectueux avec cette approche.
-   **Tâches 2 et 3 (Type et Sévérité)** : Les performances sont plus faibles (autour de 27-35%). Cela s'explique principalement par la **petite taille du jeu de données de test** (seulement 26 échantillons). Il est difficile pour le modèle de généraliser sur si peu d'exemples pour une classification multi-classes.

## 7. Conclusion et Pistes d'Amélioration

Ce notebook démontre avec succès qu'une approche basée sur l'**extraction de features statistiques et fréquentielles** à partir des données de capteurs embarqués est viable pour la détection de défauts sur les hélices de drones.

**Points Clés :**
-   La sélection rigoureuse des capteurs (gyroscopes et accéléromètres) est fondamentale.
-   Le passage du domaine temporel au domaine fréquentiel (FFT) est puissant pour identifier les vibrations.
-   L'agrégation des données par vol transforme un problème de Big Data (séries temporelles) en un problème de classification classique et gérable.

**Pistes d'Amélioration :**
1.  **Augmenter la taille du dataset** : Collecter plus de données de vol pour améliorer les performances sur les tâches de classification multi-classes.
2.  **Data Augmentation** : Appliquer des techniques pour générer artificiellement de nouvelles données d'entraînement.
3.  **Deep Learning** : Tester des modèles de deep learning (comme les CNN 1D ou les LSTM) directement sur les séries temporelles brutes, ce qui pourrait permettre de capturer des motifs complexes que les statistiques agrégées ne voient pas.

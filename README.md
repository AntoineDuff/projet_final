# Implementation of DecisionTree and NeuralNet
# Classe DecisionTree:
DecisionTree est une classe implémentant un modèle de classement suivant l'algorithme d'arbre de décision
- Le contructeur __init__() par défaut est utilisé pour cette classe sans arguments.
- train() prend les données ainsi que leur étiquette pour l'entrainement du modèle et réalise l'entrainement de ce dernier.
- predict() donne la prédiction du modèle sur une donnée quelconque.
- accuracy() retourne le score du modèle.
- confusion_matrix() prend les données test ainsi que leur étiquette en argument et retourne la matrice de confusion générée selon les données à prédires.
- test() prend les données test ainsi que leur étiquette en argument et retourne le score du modèle. Il est aussi possible d'afficher le rappel, la matrice de confusion ainsi que la précision en mettant l'argument muted à True.

# Classe NeuralNet
NeuralNet est une classe suivant l'implémentation classique du modèle de classement basé sur le perceptron multicouches.
- Le contructeur __init__() prend en paramètre le nombre de couches cachées, le type d'initialisation des poids ainsi que le nombre de neurones par couche.
- train() prend les données ainsi que leur étiquette pour l'entrainement du modèle et réalise l'entrainement de ce dernier.
- predict() donne la prédiction du modèle sur une donnée quelconque.
- accuracy() retourne le score du modèle.
- confusion_matrix() prend les données test ainsi que leur étiquette en argument et retourne la matrice de confusion générée selon les données à prédires.
- test() prend les données test ainsi que leur étiquette en argument et retourne le score du modèle. Il est aussi possible d'afficher le rappel, la matrice de confusion ainsi que la précision en mettant l'argument muted à True.

# Classe K_folds
La classe K_folds implémente la méthode de génération de données en k-plis pour la validation croisée.
- Le constructeur __init__() prend en argument le nombre de plis à utiliser.
- split() prend en arguement les données à diviser ainsi que leur étiquette. Cette méthode retourne une liste de listes des indices des données et des étiquettes à utiliser dans le jeu de donnée initiale pour chacun des plis.

# Division du travail
L'ensemble de l'implémentation des classes à été réalisée par les deux membres de l'équipe Louis-Gabriel Maltais et Antoine Dufour. Aucune difficulté particulière n'a été rencontrée lors de l'implémentation du code.

import numpy as np
import random

def load_iris_dataset(train_ratio):
    """Cette fonction a pour but de lire le dataset Iris

    Args:
        train_ratio: le ratio des exemples (ou instances) qui vont etre attribués à l'entrainement,
        le rest des exemples va etre utilisé pour les tests.
        Par exemple : si le ratio est 50%, il y aura 50% des exemple (75 exemples) qui vont etre utilisé
        pour l'entrainement, et 50% (75 exemples) pour le test.

    Retours:
        Cette fonction doit retourner 4 matrices de type Numpy, train, train_labels, test, et test_labels
		
        - train : une matrice numpy qui contient les exemples qui vont etre utilisés pour l'entrainement, chaque 
        ligne dans cette matrice représente un exemple (ou instance) d'entrainement.
		
        - train_labels : contient les labels (ou les étiquettes) pour chaque exemple dans train, de telle sorte
          que : train_labels[i] est le label (ou l'etiquette) pour l'exemple train[i]
        
        - test : une matrice numpy qui contient les exemples qui vont etre utilisés pour le test, chaque 
        ligne dans cette matrice représente un exemple (ou instance) de test.
		
        - test_labels : contient les labels (ou les étiquettes) pour chaque exemple dans test, de telle sorte
          que : test_labels[i] est le label (ou l'etiquette) pour l'exemple test[i]
    """
    
    random.seed(1) # Pour avoir les meme nombres aléatoires à chaque initialisation.
    np.random.seed(1)
    # Vous pouvez utiliser des valeurs numériques pour les différents types de classes, tel que :
    conversion_labels = {'Iris-setosa': 0, 'Iris-versicolor' : 1, 'Iris-virginica' : 2}
    
    # Le fichier du dataset est dans le dossier datasets en attaché 

    # TODO : le code ici pour lire le dataset
    X_data = []
    y_label = []
    with open('datasets/bezdekIris.data', 'r') as f:
        for line in f:
            if line == '\n':
                break
            line_val = line.strip('\n').split(',')
            temp_param = list(map(float, line_val[:4]))
            temp_label = conversion_labels[line_val[4]]
            X_data.append(temp_param)
            y_label.append(temp_label)
    
    X_data = np.asarray(X_data)
    y_label = np.asarray(y_label)

    # random (train/test) with respect to train_ratio
    indices = np.arange(X_data.shape[0])
    train_ind = int(train_ratio * X_data.shape[0])
    np.random.shuffle(indices)
    train_indices = indices[:train_ind]
    test_indices = indices[train_ind:]

    #split
    train, test = X_data[train_indices], X_data[test_indices]
    train_labels, test_labels = y_label[train_indices], y_label[test_indices]
    # REMARQUE très importante : 
	# remarquez bien comment les exemples sont ordonnés dans 
    # le fichier du dataset, ils sont ordonnés par type de fleur, cela veut dire que 
    # si vous lisez les exemples dans cet ordre et que si par exemple votre ration est de 60%,
    # vous n'allez avoir aucun exemple du type Iris-virginica pour l'entrainement, pensez
    # donc à utiliser la fonction random.shuffle pour melanger les exemples du dataset avant de séparer
    # en train et test.
       
    # Tres important : la fonction doit retourner 4 matrices (ou vecteurs) de type Numpy. 
    return (train, train_labels, test, test_labels)
	
def load_congressional_dataset(train_ratio):
    """Cette fonction a pour but de lire le dataset Congressional Voting Records

    Args:
        train_ratio: le ratio des exemples (ou instances) qui vont servir pour l'entrainement,
        le rest des exemples va etre utilisé pour les test.

    Retours:
        Cette fonction doit retourner 4 matrices de type Numpy, train, train_labels, test, et test_labels
        
        - train : une matrice numpy qui contient les exemples qui vont etre utilisés pour l'entrainement, chaque 
        ligne dans cette matrice représente un exemple (ou instance) d'entrainement.
		
        - train_labels : contient les labels (ou les étiquettes) pour chaque exemple dans train, de telle sorte
          que : train_labels[i] est le label (ou l'etiquette) pour l'exemple train[i]
        
        - test : une matrice numpy qui contient les exemples qui vont etre utilisés pour le test, chaque 
        ligne dans cette matrice représente un exemple (ou instance) de test.
		
        - test_labels : contient les labels (ou les étiquettes) pour chaque exemple dans test, de telle sorte
          que : test_labels[i] est le label (ou l'etiquette) pour l'exemple test[i]
    """
    
    # random.seed(1) # Pour avoir les meme nombres aléatoires à chaque initialisation.
    # np.random.seed(1)
    # Vous pouvez utiliser un dictionnaire pour convertir les attributs en numériques 
    # Notez bien qu'on a traduit le symbole "?" pour une valeur numérique
    # Vous pouvez biensur utiliser d'autres valeurs pour ces attributs
    conversion_labels = {'republican' : 0, 'democrat' : 1, 
                         'n' : 0, 'y' : 1, '?' : 2} 
    
    # Le fichier du dataset est dans le dossier datasets en attaché 
    # TODO : le code ici pour lire le dataset
    X_data = []
    y_label = []
    with open('datasets/house-votes-84.data', 'r') as f:
        for line in f:
            if line == '\n':
                break
            line_val = line.strip('\n').split(',')
            temp_param = list(map(lambda x: conversion_labels[x], line_val[1:]))
            temp_label = conversion_labels[line_val[0]]
            X_data.append(temp_param)
            y_label.append(temp_label)
    
    X_data = np.asarray(X_data)
    y_label = np.asarray(y_label)

    # random (train/test) with respect to train_ratio
    indices = np.arange(X_data.shape[0])
    train_ind = int(train_ratio * X_data.shape[0])
    np.random.shuffle(indices)
    train_indices = indices[:train_ind]
    test_indices = indices[train_ind:]

    #split
    train, test = X_data[train_indices], X_data[test_indices]
    train_labels, test_labels = y_label[train_indices], y_label[test_indices]
	
	# La fonction doit retourner 4 structures de données de type Numpy.
    return (train, train_labels, test, test_labels)
	
def load_monks_dataset(numero_dataset):
    """Cette fonction a pour but de lire le dataset Monks
    
    Notez bien que ce dataset est différent des autres d'un point de vue
    exemples entrainement et exemples de tests.
    Pour ce dataset, nous avons 3 différents sous problèmes, et pour chacun
    nous disposons d'un fichier contenant les exemples d'entrainement et 
    d'un fichier contenant les fichiers de tests. Donc nous avons besoin 
    seulement du numéro du sous problème pour charger le dataset.

    Args:
        numero_dataset: lequel des sous problèmes nous voulons charger (1, 2 ou 3 ?)
		par exemple, si numero_dataset=2, vous devez lire :
			le fichier monks-2.train contenant les exemples pour l'entrainement
			et le fichier monks-2.test contenant les exemples pour le test
        les fichiers sont tous dans le dossier datasets
    Retours:
        Cette fonction doit retourner 4 matrices de type Numpy, train, train_labels, test, et test_labels
        
        - train : une matrice numpy qui contient les exemples qui vont etre utilisés pour l'entrainement, chaque 
        ligne dans cette matrice représente un exemple (ou instance) d'entrainement.
        - train_labels : contient les labels (ou les étiquettes) pour chaque exemple dans train, de telle sorte
          que : train_labels[i] est le label (ou l'etiquette) pour l'exemple train[i]
        
        - test : une matrice numpy qui contient les exemples qui vont etre utilisés pour le test, chaque 
        ligne dans cette matrice représente un exemple (ou instance) de test.
        - test_labels : contient les labels (ou les étiquettes) pour chaque exemple dans test, de telle sorte
          que : test_labels[i] est le label (ou l'etiquette) pour l'exemple test[i]
    """
    if (int(numero_dataset) > 3 or int(numero_dataset) < 1):
        raise ValueError(f"Invalid dataset number. Dataset #{numero_dataset} is not a valid number.\nChoose dataset #1, #2 or #3.")
	
	# TODO : votre code ici, vous devez lire les fichiers .train et .test selon l'argument numero_dataset
    dataset_dict = {1: ['datasets/monks-1.train', 'datasets/monks-1.test'],
                    2: ['datasets/monks-2.train', 'datasets/monks-2.test'],
                    3: ['datasets/monks-3.train', 'datasets/monks-3.test']}
    dataset_X = []
    dataset_y = []
    for dataset_name in dataset_dict[int(numero_dataset)]:
        X_data = []
        y_label = []
        with open(dataset_name, 'r') as f:
            for line in f:
                if line == '\n':
                    break
                line_val = line.strip('\n').split(' ')[1:] #remove the first space of each line
                temp_param = list(map(int, line_val[1:7]))
                temp_label = int(line_val[0])
                X_data.append(temp_param)
                y_label.append(temp_label)
        
        X_data = np.asarray(X_data)
        y_label = np.asarray(y_label)
        dataset_X.append(X_data)
        dataset_y.append(y_label)
    #train position 0, test position 1
    train = dataset_X[0]
    train_labels = dataset_y[0]
    test = dataset_X[1]
    test_labels = dataset_y[1]

    # La fonction doit retourner 4 matrices (ou vecteurs) de type Numpy. 
    return (train, train_labels, test, test_labels)
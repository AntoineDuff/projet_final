import numpy as np
import matplotlib.pyplot as plt
import sys
import NeuralNet  # importer la classe du Réseau de Neurones
import DecisionTree  # importer la classe de l'Arbre de Décision
# importer d'autres fichiers et classes si vous en avez développés
# importer d'autres bibliothèques au besoin, sauf celles qui font du machine learning
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

from load_datasets import (load_congressional_dataset,
                           load_iris_dataset,
                           load_monks_dataset)

from sklearn.tree import DecisionTreeClassifier

if __name__ == "__main__":
    train, train_labels, test, test_labels = load_iris_dataset(0.7)
    # train, train_labels, test, test_labels = load_congressional_dataset(0.7)
    # train, train_labels, test, test_labels = load_monks_dataset(2)

    N_net = NeuralNet.NeuralNet(parameters_nb=4,
                                classes_nb=3,
                                hidden_layer_nb=2,
                                neurone_nb=3,
                                learning_rate=0.1)

    N_net.train(train, train_labels, epochs=3000)

    score = N_net.test(test, test_labels)
    print(score)


    # KK = MLPClassifier(hidden_layer_sizes=(3,3), activation='logistic', solver='sgd', learning_rate_init=0.1, max_iter=3000)
    # KK.fit(train, train_labels)
    # ee = KK.score(test, test_labels)


    # decision_tree_clf = DecisionTree.DecisionTree()
    # #train decision tree

    # clf = DecisionTreeClassifier()
    # clf.fit(train, train_labels)
    # OH = clf.predict(test)

    # decision_tree_clf.train(train, train_labels)

    # res = decision_tree_clf.test(test, test_labels)

    # print("ok")

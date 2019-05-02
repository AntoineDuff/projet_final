import numpy as np
import matplotlib.pyplot as plt
import sys
import NeuralNet  # importer la classe du Réseau de Neurones
import DecisionTree  # importer la classe de l'Arbre de Décision
# importer d'autres fichiers et classes si vous en avez développés
# importer d'autres bibliothèques au besoin, sauf celles qui font du machine learning

from load_datasets import (load_congressional_dataset,
                           load_iris_dataset,
                           load_monks_dataset)

class K_folds:
    def __init__(self, n_splits=5):
        self.n_splits = n_splits

    def split(self, X, y):
        # determine the minimum splittable len of the dataset for n folds 
        stop = len(X) - (len(X) % self.n_splits)
        fold_len = int(stop / self.n_splits)
        indices = np.arange(len(X))
        np.random.shuffle(indices)

        #reduce the list lenght to fit an even split
        indices = indices[:stop]
        
        train_ind_list = []
        test_ind_list = []

        for i in range(self.n_splits):
            train_ind_list.append(indices[:-fold_len])
            test_ind_list.append(indices[-fold_len:])

            indices = np.roll(indices, fold_len)

        return train_ind_list, test_ind_list

if __name__ == "__main__":
    train, train_labels, test, test_labels = load_iris_dataset(0.7)
    # train, train_labels, test, test_labels = load_congressional_dataset(0.7)
    # train, train_labels, test, test_labels = load_monks_dataset(3)

    "Decision tree learning curve with cross-validation"
    # All considered training dataset sizes
    # train_sizes = list(range(5, len(train[:,1]), 5))
    # avg_scores = []
    # for size in train_sizes: # For every training dataset sizes, do cross-validation
        # decision_tree_clf = DecisionTree.DecisionTree()
        # kf = K_folds(n_splits=5) 
        # train_kf, train_label_kf = kf.split(train[:size], train_labels[:size])
        # avg_score = 0
        # for train_inds, test_inds in zip (train_kf, train_label_kf):
        #     X_train = train[train_inds]
        #     y_train = train_labels[train_inds]
        #     X_test = train[test_inds]
        #     y_test = train_labels[test_inds]

        #     decision_tree_clf.train(X_train, y_train)
        #     avg_score += decision_tree_clf.test(X_test, y_test)

    #     avg_score = avg_score / kf.n_splits
    #     avg_scores.append(avg_score)

    # plt.plot(train_sizes, avg_scores)
    # plt.xlabel("Taille de l'échantillon d'entraînement")
    # plt.ylabel('Proportion correctement prédite sur le test')
    # plt.show()

    " Train and test with entire datasets, print accuracy and confusion mat."
    # decision_tree_clf = DecisionTree.DecisionTree()
    # decision_tree_clf.train(train, train_labels)
    # decision_tree_clf.test(test, test_labels, muted=False)

    " Train and test a neural network "
    N_net = NeuralNet.NeuralNet(parameters_nb=np.size(train,1),
                                classes_nb=np.size(np.unique(train_labels)),
                                hidden_layer_nb=1,
                                neurone_nb=7,
                                learning_rate=0.1)

    N_net.train(train, train_labels, epochs=500)
    score = N_net.test(test, test_labels)
    print(score)

    # print(np.size(train,1))
    # print(np.size(np.unique(train_labels)))

    " Cross-validation for best number of neurons in hidden layer."
    # kf = K_folds(n_splits=5)
    # train_kf, train_label_kf = kf.split(train, train_labels)
    # n_neurons = [4,6,8,12,16,20,24,30,40,50]
    'Pour éviter bug numpy'
    # for n in n_neurons:
    #     if n == np.size(train,1):
    #         idx = n_neurons.index(n)
    #         n_neurons[idx] = n + 1

    # avg_scores = list()
    # for n in n_neurons:
    #     N_net = NeuralNet.NeuralNet(parameters_nb=np.size(train,1),
    #                             classes_nb=np.size(np.unique(train_labels)),
    #                             hidden_layer_nb=1,
    #                             neurone_nb=n,
    #                             learning_rate=0.1)
    #     avg_score = 0
    #     for train_inds, test_inds in zip (train_kf, train_label_kf):
    #         X_train = train[train_inds]
    #         y_train = train_labels[train_inds]
    #         X_test = train[test_inds]
    #         y_test = train_labels[test_inds]

    #         N_net.train(X_train, y_train, epochs=500)
    #         avg_score += N_net.test(X_test, y_test)
        
    #     avg_score = avg_score / kf.n_splits
    #     avg_scores.append(avg_score)

    # plt.plot(n_neurons, avg_scores)
    # plt.xlabel('Nombre de neurones dans la couche cachée')
    # plt.ylabel('Accuracy moyenne sur la validation croisée')
    # plt.show()
    # print(avg_scores)

    " Cross-validation for best number of hidden layers."
    # train_sizes = list(range(10, len(train[:,1]), 10))
    # n_hidden_layers = list(range(3, 8))

    # n_neurons = 6 # Iris
    # n_neurons = 17 # Congress
    # n_neurons = 20 # MONKS-1
    # n_neurons = 24 # MONKS-2
    # n_neurons = 40 # MONKS-3

    # all_avg_scores = list()
    # for n in n_hidden_layers:
    #     avg_scores = list()
    #     for size in train_sizes: # For every training dataset sizes, do cross-validation
    #         N_net = NeuralNet.NeuralNet(parameters_nb=np.size(train,1),
    #                                 classes_nb=np.size(np.unique(train_labels)),
    #                                 hidden_layer_nb=n,
    #                                 neurone_nb=n_neurons,
    #                                 learning_rate=0.1)
    #         kf = K_folds(n_splits=5) 
    #         train_kf, train_label_kf = kf.split(train[:size], train_labels[:size])
    #         avg_score = 0
    #         for train_inds, test_inds in zip (train_kf, train_label_kf):
    #             X_train = train[train_inds]
    #             y_train = train_labels[train_inds]
    #             X_test = train[test_inds]
    #             y_test = train_labels[test_inds]

    #             N_net.train(X_train, y_train, epochs=500)
    #             avg_score += N_net.test(X_test, y_test)
      
    #         avg_score = avg_score / kf.n_splits
    #         avg_scores.append(avg_score)

    #     all_avg_scores.append(avg_scores)

    # for _, scores in enumerate(all_avg_scores):
    #     plt.plot(train_sizes, scores)
    # plt.gca().legend(('RN-3C', 'RN-4C', 'RN-5C', 'RN-6C', 'RN-7C'))
    # plt.xlabel("Taille de l'échantillon d'entraînement")
    # plt.ylabel('Proportion correctement prédite sur le test')
    # plt.show()

    " Test initialize weights to zero or random between -0.05-0.05."
    n_neurons = 6 # Iris
    n_hidden_layers = 1 # Iris
    # n_neurons = 17 # Congress
    # n_hidden_layers = 1 # Congress
    # n_neurons = 20 # MONKS-1
    # n_hidden_layers = 1 # MONKS-1
    # n_neurons = 24 # MONKS-2
    # n_hidden_layers = 1 # MONKS-2
    # n_neurons = 40 # MONKS-3
    # n_hidden_layers = 1 # MONKS-3

    all_avg_scores = list()

    # Non-zero weights
    avg_scores = list()
    train_sizes = list(range(10, len(train[:,1]), 10))
    for size in train_sizes: # For every training dataset sizes, do cross-validation
        N_net = NeuralNet.NeuralNet(parameters_nb=np.size(train,1),
                                classes_nb=np.size(np.unique(train_labels)),
                                hidden_layer_nb=n_hidden_layers,
                                neurone_nb=n_neurons,
                                learning_rate=0.1)
        kf = K_folds(n_splits=5) 
        train_kf, train_label_kf = kf.split(train[:size], train_labels[:size])
        avg_score = 0
        
        for train_inds, test_inds in zip (train_kf, train_label_kf):
            X_train = train[train_inds]
            y_train = train_labels[train_inds]
            X_test = train[test_inds]
            y_test = train_labels[test_inds]

            N_net.train(X_train, y_train, epochs=500)
            avg_score += N_net.test(X_test, y_test)
      
        avg_score = avg_score / kf.n_splits
        avg_scores.append(avg_score)

    all_avg_scores.append(avg_scores)

    # Zero weights
    avg_scores = list()
    for size in train_sizes:
        N_net = NeuralNet.NeuralNet(parameters_nb=np.size(train,1),
                                classes_nb=np.size(np.unique(train_labels)),
                                hidden_layer_nb=n_hidden_layers,
                                neurone_nb=n_neurons,
                                learning_rate=0.1, zero_weights=True)
        kf = K_folds(n_splits=5) 
        train_kf, train_label_kf = kf.split(train[:size], train_labels[:size])
        avg_score = 0
        for train_inds, test_inds in zip (train_kf, train_label_kf):
            X_train = train[train_inds]
            y_train = train_labels[train_inds]
            X_test = train[test_inds]
            y_test = train_labels[test_inds]

            N_net.train(X_train, y_train, epochs=500)
            avg_score += N_net.test(X_test, y_test)
      
        avg_score = avg_score / kf.n_splits
        avg_scores.append(avg_score)

    all_avg_scores.append(avg_scores)

    for _, scores in enumerate(all_avg_scores):
        plt.plot(train_sizes, scores)
    plt.gca().legend(('RN-NON-ZERO', 'RN-ZERO'))
    plt.xlabel("Taille de l'échantillon d'entraînement")
    plt.ylabel('Proportion correctement prédite sur le test')
    plt.show()
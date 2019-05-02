"""
Vous allez definir une classe pour chaque algorithme que vous allez développer,
votre classe doit contenit au moins les 3 methodes definies ici bas, 
	* train 	: pour entrainer le modèle sur l'ensemble d'entrainement
	* predict 	: pour prédire la classe d'un exemple donné
	* test 		: pour tester sur l'ensemble de test
vous pouvez rajouter d'autres méthodes qui peuvent vous etre utiles, mais moi
je vais avoir besoin de tester les méthodes train, predict et test de votre code.
"""
import numpy as np

#additionnal libraries
import matplotlib.animation as animation
import matplotlib.pyplot as plt

def scale_mat(mat, a, b):
	scaled_result = (mat * (b-a)) + a

	return scaled_result

class NeuralNet:

	def __init__(self, parameters_nb, classes_nb, hidden_layer_nb, neurone_nb, learning_rate=0.01,
				zero_weights=False):
		"""
		c'est un Initializer. 
		Vous pouvez passer d'autre paramètres au besoin,
		c'est à vous d'utiliser vos propres notations
		"""
		#define class values
		self.layers_weights = []
		self.bias = []
		self.classes_nb = classes_nb
		self.learning_rate = learning_rate
		self.neurone_nb = neurone_nb
		
		#initialize weights
		for i in range(hidden_layer_nb + 1):
			if i == 0:
				if zero_weights == True:
					result = np.zeros((parameters_nb, neurone_nb))
				else:
					weights = np.random.rand(parameters_nb, neurone_nb)
					result = scale_mat(weights, -0.05, 0.05)
				self.layers_weights.append(result)
				self.bias.append(np.ones(neurone_nb))

			elif i == hidden_layer_nb:
				if zero_weights == True:
					result = np.zeros((neurone_nb, classes_nb))
				else:
					weights = np.random.rand(neurone_nb, classes_nb)
					result = scale_mat(weights, -0.05, 0.05)
				self.layers_weights.append(result)
				self.bias.append(np.ones(classes_nb))
				
			else:
				if zero_weights == True:
					result = np.zeros((neurone_nb, neurone_nb))
				else:
					weights = np.random.rand(neurone_nb, neurone_nb)
					result = scale_mat(weights, -0.05, 0.05)
				self.layers_weights.append(result)
				self.bias.append(np.ones(neurone_nb))

		self.layers_weights = np.array(self.layers_weights)

	def train(self, train, train_labels, epochs=1000, show_graph=False):
		"""
		c'est la méthode qui va entrainer votre modèle,
		train est une matrice de taille nxm, avec 
		n : le nombre d'exemple d'entrainement dans le dataset
		m : le mobre d'attribus (le nombre de caractéristiques)
		
		train_labels : est une matrice de taille nx1
		
		vous pouvez rajouter d'autres arguments, il suffit juste de
		les expliquer en commentaire
		------------
		"""
		from tqdm import tqdm
		scores = []

		for i in tqdm(range(epochs)):
			#shuffle dataset every epoch
			X, Y = self._dataset_shuffle(train, train_labels)
			for x, y in zip(X, Y):
				pred, X_list, O_list = self._forward(x)
				self._backward_propagation(X_list, O_list, pred, y)

			score = self.test(train, train_labels)
			scores.append(score)
		if show_graph == True:
			plt.plot(np.arange(epochs), scores)
			plt.show()

	def predict(self, exemple, label):
		"""
		Prédire la classe d'un exemple donné en entrée
		exemple est de taille 1xm
		
		si la valeur retournée est la meme que la veleur dans label
		alors l'exemple est bien classifié, si non c'est une missclassification

		"""
		preds_h, _, _ = self._forward(exemple)

		return np.argmax(preds_h)

	def test(self, test, test_labels):
		"""
		c'est la méthode qui va tester votre modèle sur les données de test
		l'argument test est une matrice de taille nxm, avec 
		n : le nombre d'exemple de test dans le dataset
		m : le mobre d'attribus (le nombre de caractéristiques)
		
		test_labels : est une matrice taille nx1
		
		vous pouvez rajouter d'autres arguments, il suffit juste de
		les expliquer en commentaire
		
		Faites le test sur les données de test, et afficher :
		- la matrice de confision (confusion matrix)
		- l'accuracy (ou le taux d'erreur)
		
		Bien entendu ces tests doivent etre faits sur les données de test seulement
		
		"""
		preds = []
		for x, y in zip(test, test_labels):
			pred = self.predict(x, y)
			preds.append(pred)

		is_equal = np.zeros_like(test_labels, dtype=bool)
		is_equal[preds == test_labels] = True

		correct_preds_count = np.count_nonzero(is_equal)
		score = correct_preds_count / len(test_labels)

		return score

	def _forward(self, x):
		X = x.copy()
		O_list = []
		X_list = [X]
		#forward pass for every layer
		for single_layer_weight, layer_bias in zip(self.layers_weights, self.bias):
			X = np.matmul(X, single_layer_weight)
			#add bias
			# X = X + layer_bias
			X_list.append(X.copy())
			#apply activation function
			O = self._sigmoid(X)
			#append neurones values
			O_list.append(O.copy())
			X = O

		preds_h = X.copy()

		return preds_h, X_list, O_list

	def _backward_propagation(self, X_list, O_list, preds_h, label):

		prev_weights = self.layers_weights.copy()
		one_hot = np.zeros(self.classes_nb)
		one_hot[label] = 1

		#update output layer weights
		# delta_k = -preds_h  *  (one_hot - preds_h) * self._deriv_sigmoid(preds_h)
		delta_k = preds_h * (1 - preds_h) * (one_hot - preds_h)

		delta_k = np.reshape(delta_k, (1, self.classes_nb))
		X_last = X_list[-2]
		X_last = np.reshape(X_last, (self.neurone_nb, 1))

		self.layers_weights[-1] += self.learning_rate * np.matmul(X_last, delta_k)
		# self.bias[-1] += self.learning_rate * np.reshape(delta_k, self.classes_nb)
		
		delta_k = np.transpose(delta_k)

		#update hidden layers weights
		for O_h, W, X, layer_weights in zip(reversed(O_list[:-1]),
											reversed(prev_weights[1:]),
											reversed(X_list[:-2]),
											reversed(self.layers_weights[:-1])):
			delta = O_h * (1 - O_h)
			dot = np.dot(W, delta_k)
			delta_h = delta[:, None] * dot

			temp_delta_h = np.reshape(delta_h, (1, delta_h.shape[0]))
			X = np.reshape(X, (X.shape[0], 1))

			delta_w = self.learning_rate * np.matmul(X, temp_delta_h)
			layer_weights += delta_w
			delta_k = delta_h

	def _sigmoid(self, x):

		return 1 / (1 + np.exp(-x))

	def _dataset_shuffle(self, data, labels):
		#shuffled indices
		indices = np.arange(len(data))
		np.random.shuffle(indices)

		shuffled_data = data[indices]
		shuffled_labels = labels[indices]

		return shuffled_data, shuffled_labels
import random
import sys
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

class DecisionTree:

	def __init__(self, **kwargs):
		"""
		Init
		"""

	def _plurality_value(self, examples):
		classes, classes_item_count = np.unique(examples, return_counts=True)
		arg_max = np.argmax(classes_item_count)
		max_val = classes_item_count[arg_max]

		is_equal = np.zeros_like(classes_item_count, dtype=bool)
		#if more than one element is maximal set to true
		is_equal[classes_item_count == max_val] = True
		#pick the most communs classes if tie
		most_common_classes = classes[is_equal == True]
		#randomly select from possible class values if more than one
		result_class = random.choice(most_common_classes)
		
		return result_class

	def _decision_tree_learning(self, examples, classes, parent, parent_classes, tree, node, attr_ind_list):
		#examples is empty
		if len(examples) == 0:
			plural_val = self._plurality_value(parent_classes)
			tree.add_node(plural_val)
			tree.add_edge(node, plural_val)

			return None

		# example are from the same class
		elif len(np.unique(classes)) == 1:
			tree.add_node(classes[0])
			tree.add_edge(node, classes[0])

			return None

		#no attributes left
		elif len(examples[0]) == 0:
			plural_val = self._plurality_value(classes)
			tree.add_node(plural_val)
			tree.add_edge(node, plural_val)

			return None

		else:
			best_attr_idx = self._importance(examples, parent_classes, classes)
			best_attr = examples[:, best_attr_idx]
			#TODO maybe tune continous values to range
			possible_val_attr = np.unique(best_attr)

			sub_examples_set = self._slice_attribute(examples, best_attr_idx)

			#real attribute index for node
			real_idx = attr_ind_list[best_attr_idx]

			sub_attr_ind_list = np.hstack((attr_ind_list[0:best_attr_idx], attr_ind_list[best_attr_idx+1:])).copy()

			for pos_val in possible_val_attr:
				sub_node = (pos_val, real_idx, np.random.random())
				#TODO  SUBNODE can be overwrite
				tree.add_node(sub_node)
				tree.add_edge(node, sub_node)
				
				indices = np.arange(len(examples))[examples[:, best_attr_idx] == pos_val]

				sub_examples = sub_examples_set[indices]
				sub_classes = classes[indices]

				self._decision_tree_learning(sub_examples, sub_classes, examples, classes, tree, sub_node, sub_attr_ind_list)


	def _slice_attribute(self, data, idx):
		result = np.hstack((data[:, 0:idx], data[:, idx+1:]))

		return result

	def train(self, train, train_labels):
		#create decision tree
		self.classes, self.classes_count = np.unique(train_labels, return_counts=True)

		decision_tree = nx.DiGraph()
		decision_tree.add_node('start')

		self.decision_tree = decision_tree

		first_attr_idx = self._importance(train, train_labels, train_labels)
		first_attr = train[:, first_attr_idx]
		possible_val_attr = np.unique(first_attr)

		sub_train_set = self._slice_attribute(train, first_attr_idx)

		attr_ind_list = np.arange(len(train[0]))
		attr_ind_list = np.hstack((attr_ind_list[0:first_attr_idx], attr_ind_list[first_attr_idx+1:]))

		for pos_val in possible_val_attr:
			node = (pos_val, first_attr_idx)
			decision_tree.add_node(node)
			decision_tree.add_edge('start', node)
			
			indices = np.arange(len(train))[train[:, first_attr_idx] == pos_val]

			examples = sub_train_set[indices]
			classes = train_labels[indices]

			self._decision_tree_learning(examples, classes, train, train_labels, decision_tree, node, attr_ind_list)

	def predict(self, exemple, label):
		"""
		Prédire la classe d'un exemple donné en entrée
		exemple est de taille 1xm
		
		si la valeur retournée est la meme que la veleur dans label
		alors l'exemple est bien classifié, si non c'est une missclassification

		"""
		start_succ = list(self.decision_tree.successors('start'))
		attr_ind = start_succ[0][1]
		value_eval = exemple[attr_ind]
		if self.decision_tree.has_node((value_eval, attr_ind)) == False:
			arg_max = np.argmax(self.classes_count)
			return self.classes[arg_max]
			
		succs = list(self.decision_tree.successors((value_eval, attr_ind)))
		if type(succs[0]) == tuple:
			attr_ind = succs[0][1]

		while True:
			if len(succs) == 1 and type(succs[0]) is not tuple:
				pred_class = succs[0]
				break

			else:
				temp_succs = None
				for succ_val in succs:
					if succ_val[0] == exemple[attr_ind]:
						succ_name = succ_val
						temp_succs = list(self.decision_tree.successors(succ_name))
						break

				if temp_succs == None:

					predecessor = list(self.decision_tree.predecessors(succs[0]))
					source = predecessor[0]
					count_oc_class = []
					for i_class in self.classes:
						# try:
						if nx.has_path(self.decision_tree, source, int(i_class)):
							pred_cl = list(nx.all_shortest_paths(self.decision_tree, source, int(i_class)))
							count_oc_class.append(len(pred_cl))
						else:
							count_oc_class.append(0)
					
					arg_max = np.argmax(count_oc_class)

					return self.classes[arg_max]
				succs = temp_succs
				if type(temp_succs[0]) == tuple:
					attr_ind = temp_succs[0][1]

		return pred_class
					
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
		- la matrice de confusion (confusion matrix)
		- l'accuracy (ou le taux d'erreur)
		
		Bien entendu ces tests doivent etre faits sur les données de test seulement
		
		"""
		preds = []
		for test_val, test_label in zip(test, test_labels):
			temp_pred_val = self.predict(test_val, test_label)

			preds.append(temp_pred_val)

		matching_preds = np.zeros_like(preds, dtype=bool)
		matching_preds[preds == test_labels] = True
		_, bad_good_preds  = np.unique(matching_preds, return_counts=True)

		score = bad_good_preds[1] / len(matching_preds)

		return score
	
	def _compute_entropy(self, classes):
		_, class_count = np.unique(classes, return_counts=True)
		total = np.sum(class_count)

		entropy = -1 * np.sum((class_count * (1 / total)) * np.log2(class_count * (1 / total)))

		return entropy

	## parent = single attribute
	def _importance(self, example, parent_classes, labels):
		"""
		Compute the information gain
		Return indice of the attribute with max gain
		"""
		#compute parent entropy
		parent_entropy = self._compute_entropy(parent_classes)

		nb_data, nb_attr = example.shape
		gain_attr_list = []
		for i in range(nb_attr):
			possible_val, attr_count = np.unique(example[:, i], return_counts=True)
			total_count = np.sum(attr_count)
			entropy = 0
			for poss in possible_val:
				class_ind = np.arange(nb_data)[example[:, i] == poss]
				associated_classes = labels[class_ind]
				entropy += (len(associated_classes) / total_count) * \
							self._compute_entropy(associated_classes)

			attr_gain = parent_entropy - entropy
			gain_attr_list.append(attr_gain)

		max_gain_ind = np.argmax(gain_attr_list)

		return max_gain_ind

	# Vous pouvez rajouter d'autres méthodes et fonctions,
	# il suffit juste de les commenter.
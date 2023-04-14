"""
The :mod:`sblearn.models` module implements symbolic regression. Symbolic regression 
is a supervised learning model that uses evolutionary algorithms to find
mathematical relationships in the data.
"""

# Author: Vincent Papelard <papelardvincent@gmail.com>
#
# License: MIT

import multiprocessing
import random
from copy import deepcopy

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.exceptions import NotFittedError
from sklearn.metrics import r2_score
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

import sblearn.compute as compute
import sblearn.trees as trees


class SymbolicRegressor(BaseEstimator, RegressorMixin):
	"""
	A symbolic regression estimator based on sklearn's 
	`BaseEstimator` and `RegressorMixin` classes.
	As such, it has all the features of any other sklearn
	regression estimator (GridSearchCV, pipelines and so on.)

	Symbolic regression is a type of regression model that 
	combines mathematical blocks to find the function that best 
	fits the data. Here each function is represented as a binary 
	tree.

	Parameters
		population_size (int, default=10000):
			The size of the functions population
			to use in each generation of the evolutionary algorithm.
			Increasing this often yields better results, but also significantly
			increases training time.

		n_iter (int, default=20):
			The number of successive generations your algorithm
			must have. Increasing this sometimes leads to better performance
			but also increases training time.

		sampling_rate (float, default=0.15, ]0,1] value):
			In order to improve the model's generalization 
			capabilities, each generation is trained on a random subset of the 
			training dataset only. The sampling rate defines how much of the 
			whole set is used for training at each generation. While it is
			generally not advised to change this value, decreasing it might
			be a good idea if your model is overfitting.

		mutation_chance (float, default=0.3, ]0,1] value):
			The proportion of function trees that mutate
			at each generation. It is advised to leave the default value.

		elitism (bool, default=True):
			If set to True, the best functions in a generation will
			always be selected as-it-is for the next generation. The functions
			defined as "elits" are the best 2%.

			*Note*: if set to False, parameters `adjust_elites` and `adjust_range` are disabled.

		adjust_elites (bool, default=True): 
			If set to True, at each generation new functions will
			be created based on elites functions. These new functions will be multiplied 
			by 1 + a random float in `(-constants_range, constants_range)`. This avoids 
			being stuck in a local optimum by applying small changes to the most promising 
			functions.

		adjust_range (float, default=0.15, ]0, 1] value):
			Sets the range from which the random adjustment coefficients
			can be drawn.

		replacement_rate (float, default=0, ]0, 1] value): 
			Sets the proportion of functions that will be replaced
			by new, randomnly-generated functions at each generation. Elite functions cannot
			be replaced.

		max_depth (int/str, default='auto'): 
			Sets the max depth function trees can have. A higher max depth means
			a longer training and might lead to overfitting, so it is better to keep that parameter
			low.
			The default value is 'auto', meaning that the max depth is equal to the number of features + 2.

		simpler_is_better (bool, default=False): 
			If set to True, the fitness function used during training takes into
			account the function's complexity, meaning that more complex functions are penalized based
			on their complexity using a parsimony coefficient. This ensures that the fitted function 
			will be easily readable and can sometimes avoid overfitting by avoiding bloating phenomena,
			but it reduces performances in most situations.

		constants_range (str/tuple/list, default='auto'):
			Defines the range from which random constant values used in function
			trees are generated. It is better to generate values on the same order of magnitude as
			the data contained in our dataset. The 'auto' value uses as range :math:`[p1, p2]`, with
			p1 and p2 the 5th and 95th percentiles of all our features data taken together.

		verbose (int, default=0):
			Defines how much information is displayed during training.
			0: nothing is displayed
			1: the fitted functions' simplified expressions are displayed at the end of training
			2: average fitness is displayed for each generation

		random_state (int/None, default=None):
			Sets the random seed to use for reproducibility.

		n_jobs (int, default=-1):
			The number of cores to use in parallel during training. If sets to -1, all available
			cores are used.

	Attributes:
		formulas (list):
			A list of the simplified math expressions estimated for each target value stored 
			as strings.

		trees (list):
			A list of tree representations stored as strings for each target value.

	Example::
		
		>>> from sblearn.models import SymbolicRegressor
		>>> model = SymbolicRegressor()
		>>> model.fit(X_train, y_train)
		>>> print(model.formulas)
		['y0 = 21.291046142578125*x0 + 47.842154502868652']
	"""

	def __init__(
		self, 
		population_size=10000,
		n_iter=20,
		sampling_rate=0.15,
		mutation_chance=0.3,
		elitism=True,
		adjust_elites=True,
		adjust_range=0.15,
		replacement_rate=0,
		max_depth='auto',
		simpler_is_better=False,
		constants_range='auto',
		verbose=0,
		random_state=None,
		n_jobs=-1
		):

		self.population_size = population_size
		self.n_iter = n_iter
		self.sampling_rate = sampling_rate
		self.mutation_chance = mutation_chance
		self.elitism = elitism
		self.adjust_elites = adjust_elites
		self.adjust_range = adjust_range
		self.replacement_rate = replacement_rate
		self.max_depth = max_depth
		self.simpler_is_better = simpler_is_better
		self.constants_range = constants_range
		self.verbose = verbose
		self.random_state = random_state
		self.n_jobs = n_jobs

	@property
	def formulas(self):
		check_is_fitted(self)
		return self.__formulas
	
	@formulas.setter
	def formulas(self, new_value):
		raise AttributeError("Cannot set the fitted function manually, run fit method instead")
	
	@property
	def trees(self):
		check_is_fitted(self)
		return self.__trees

	@trees.setter
	def trees(self, new_value):
		raise AttributeError("Cannot set the fitted function manually, run fit method instead")


	def fit(self,
		X, 
		y,
		):
		"""
		Trains the model with the data provided and returns a trained `SymbolicRegressor` instance.

		:param X: Training data.
		:type X: array-like of shape (n_samples, n_features)

		:param y: Target values.
		:type y: array-like of shape (n_samples,) or (n_samples, n_targets)

		:return: self
		:rtype: SymbolicRegressor
		"""

		if self.n_jobs == -1: 
			processes_count = multiprocessing.cpu_count()
		elif self.n_jobs == 0:
			processes_count = 1
		else: 
			processes_count = self.n_jobs
			
		if not self.random_state:
			self.current_random_state = random.randint(0, 10000)
		else:
			self.current_random_state = self.random_state
			random.seed(self.current_random_state)
			np.random.seed(self.current_random_state)
			trees.set_random_state(self.current_random_state)
			
		self.elitism_rate = 0.02
		self.__fitted_function = None

		# Check that X and y have correct shape
		X, y = check_X_y(X, y)
		
		try: 
			self.input_dim = X.shape[1]
		except IndexError: 
			self.input_dim = 1
		try: 
			self.output_dim = y.shape[1]
		except IndexError: 
			self.output_dim = 1

		if self.constants_range == 'auto':
			self.new_constants_range = (
				np.percentile(X, 5), 
				np.percentile(X, 95)
				)

		else: 
			self.new_constants_range = self.constants_range

		if self.max_depth == 'auto': 
			self.tree_depth = self.input_dim + 2
		else: 
			self.tree_depth = self.max_depth

		X = X.copy()


		if self.simpler_is_better:
			self.parsimony_coef = np.mean(y) * 0.05
		else: 
			self.parsimony_coef = 0.0
		
		self.min_float = np.finfo(np.float32).min
		self.max_float = np.finfo(np.float32).max

		X = X.astype('float32')
		y = y.astype('float64')
		
		for gen in range(self.n_iter):

			idx = np.random.choice(
				np.arange(len(X)), 
				int(self.sampling_rate*len(X)), 
				replace=False
				)
			
			x_sample = X[idx]
			y_sample = y[idx]


			if gen == 0: 
				pop = self.__generate_random_population(self.population_size)
			else:
				pop = self.__select_next_generation(pop)

			all_df = np.array_split(pop, processes_count)
			output = Parallel(n_jobs=processes_count)(delayed(
				compute.compute_functions_result)
				(df, x_sample, y_sample, self.output_dim, self.min_float, self.max_float, self.parsimony_coef) for df in all_df)

			pop = pd.concat(output, ignore_index=True, axis=0)


			average_perf = pop["perf"].mean()
			if average_perf == np.inf: avg = self.max_float
			elif average_perf == np.NINF: avg = self.min_float
			elif average_perf == np.nan: avg = self.max_float
			else: avg = average_perf

			if self.verbose >= 2: 
				print(f"Generation {gen} - Average fitness: {avg}")

		all_df = np.array_split(pop, processes_count)
		output = Parallel(n_jobs=processes_count)(delayed(
			compute.compute_functions_result)
			(df, X, y, self.output_dim, self.min_float, self.max_float, self.parsimony_coef) for df in all_df)
		
		pop = pd.concat(output, ignore_index=True, axis=0)

		best_function = pop.iloc[pop["perf"].idxmin()]
		self.__fitted_function = best_function
		self.__formulas = []
		self.__trees = []
		for dim in range(self.output_dim):
			self.__formulas.append(f"y{dim} = {best_function[f'tree{dim}'].get_expression()}")
			self.__trees.append(f"y{dim}_tree: {best_function[f'tree{dim}'].get_tree()}")

		if self.verbose >= 1:
			print("BEST SOLUTION:")
			for dim in range(self.output_dim):
				print(self.__formulas[dim])

		self.is_fitted_ = True
		return self


	def predict(
			self, 
			X
			):
		"""
		Predicts using the model.

		:param X: Samples to use to make predictions.
		:type X: array-like of shape (n_samples, n_features)
		"""

		check_is_fitted(self)
		X_ = check_array(X)
		X_ = X_.astype('float32')
		y_pred = pd.DataFrame()
		function = self.__fitted_function

		for dim in range(self.output_dim):
			output_tree = function[f"tree{str(dim)}"]
			dim_pred = compute.predict_result(output_tree, X_)
			y_pred[f"y{str(dim)}"] = dim_pred

		np.nan_to_num(y_pred, copy=False, posinf=self.max_float, neginf=self.min_float)
		y_pred = np.squeeze(y_pred)
		return y_pred.values

	def score(
			self, 
			X, 
			y
			):
		"""
		Return the coefficient of determination (R2) of the prediction.

		:param X: Test samples
		:type X: array-like of shape (n_samples, n_features)

		:param y: True values for X.
		:type y: array-like of shape (n_samples,) or (n_samples, n_targets)

		"""
		y_pred = self.predict(X.astype('float32'))
		return r2_score(y_pred, y)
	

	def __generate_random_population(
			self, 
			pop_size
			):
		"""
		Generates a random population of functions.
		"""
		
		pop = []
		for i in range(int(pop_size)): 
			pop.append(self.__generate_individual())
		pop = pd.DataFrame(pop)
		return pop

	def __generate_individual(self):
		
		"""
		Generates a single individual, i.e. a set of functions, where each function 
		predicts the output for a target.
		"""

		functions = {}
		for output in range(self.output_dim):
			functions[f"tree{output}"] = trees.generate_function(
				self.tree_depth, 
				self.input_dim, 
				self.new_constants_range
				)
		return functions

	def __select_next_generation(
			self, 
			pop
			):
		"""
		Applies genetic programming to select the population for the next generation.
		"""

		parents_among_best = 0.8
		pop.sort_values(by="perf", inplace=True)

		########################################################################
		# ELITISM & ADJUSTMENT
		########################################################################
		if self.elitism: 
			elites = pop.iloc[:int(self.elitism_rate * self.population_size)]

			elites = pd.DataFrame(deepcopy(elites.to_dict())) 
			elites.drop(columns=["perf"], inplace=True)

			if self.adjust_elites:
				adjusted_elites = pd.DataFrame()
				for dim in range(self.output_dim):
					adjusted_elites[f"tree{str(dim)}"] = elites[f"tree{str(dim)}"].apply(
						lambda x: trees.adjust(x, self.adjust_range)
						)
				elites = pd.concat([elites, adjusted_elites], ignore_index=True, axis=0)


		########################################################################
		# CROSSOVER
		########################################################################
		best_individuals = pop.iloc[:self.population_size//2]
		selected_best = best_individuals.sample(
			round(parents_among_best * len(best_individuals)), 
			random_state=self.current_random_state
			)
		
		worst_individuals = pop.iloc[self.population_size//2:]
		selected_worst = worst_individuals.sample(
			round((1 - parents_among_best) * len(worst_individuals)), 
			random_state=self.current_random_state
			)

		selected_parents = pd.concat([selected_best, selected_worst], ignore_index=True, axis=0)
		selected_parents = selected_parents.sample(frac=1, random_state=self.current_random_state)
		selected_parents.drop(columns=["perf"], inplace=True)
		new_pop = deepcopy(selected_parents)

		parents_df = np.array_split(selected_parents, 2)
		if len(parents_df[0]) > len(parents_df[1]): 
			parents_df[0].drop(parents_df[0].tail(1).index, inplace=True) 
		children_df = pd.DataFrame()

		for dim in range(self.output_dim):

			p1 = parents_df[0][f"tree{str(dim)}"]
			p2 = parents_df[1][f"tree{str(dim)}"]
			parents = pd.DataFrame(p1.values, columns=["parents_a"])
			parents["parents_b"] = p2.values
			
			dim_children = parents.apply(
				lambda row: trees.cross_functions(
					row["parents_a"], 
					row["parents_b"]), 
					axis=1, 
					result_type='expand'
				)
			children_df[f"tree{str(dim)}"] = pd.concat(
				[dim_children[0], dim_children[1]], 
				ignore_index=True, axis=0
				)

		new_pop = pd.concat([new_pop, children_df], ignore_index=True, axis=0)


		########################################################################
		# RANDOM FUNCTIONS GENERATION
		########################################################################
		new_individuals_count = int(self.replacement_rate * len(new_pop))
		new_individuals = self.__generate_random_population(new_individuals_count)
		new_pop = new_pop.sample(
			len(new_pop) - len(new_individuals), 
			random_state=self.current_random_state
			)
		new_pop = pd.concat([new_pop, new_individuals], ignore_index=True, axis=0)


		########################################################################
		# MUTATION
		########################################################################
		trees_to_change = int(self.mutation_chance * len(new_pop.index.values) // 1)
		rows_to_mutate = new_pop.sample(trees_to_change, random_state=self.current_random_state)

		for dim in range(self.output_dim):
			rows_to_mutate[f"tree{str(dim)}"] = rows_to_mutate[f"tree{str(dim)}"].apply(
				lambda x: trees.mutate(x, input_dim=self.input_dim, constants_range=self.new_constants_range)
				)
		new_pop.update(rows_to_mutate)

		if self.elitism:
			if len(new_pop) + len(elites) > self.population_size: 
				new_pop = new_pop.sample(
					self.population_size-len(elites), 
					random_state=self.current_random_state
					)

			new_pop = pd.concat([elites, new_pop], ignore_index=True, axis=0)

		new_pop.drop_duplicates(inplace=True)


		########################################################################
		# RANDOM FUNCTIONS GENERATION
		########################################################################
		new_functions = self.__generate_random_population(self.population_size-len(new_pop))
		new_pop = pd.concat([new_pop, new_functions], ignore_index=True, axis=0)

		return new_pop

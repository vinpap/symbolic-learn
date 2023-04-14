"""
Contains convenience functions to compute functions outputs using the trees 
objects defined in trees.pyx.
The cpdef functions are often intended to act as interfaces between the
pure-Python and Cython parts of the code, and are the only ones callable
from Python.
"""

# Author: Vincent Papelard <papelardvincent@gmail.com>
#
# License: MIT

import numpy as np
from sklearn.metrics import mean_absolute_error

cimport numpy as np
np.import_array()

cpdef np.ndarray predict_result(
        function, 
        np.ndarray[np.float32_t, ndim=2] X
        ):
        """
        Predicts the results over a single X array using the function
        passed as a parameter. This function is mostly here so it acts 
        as an interface between the pure-Python and Cython parts of the
        code.
        """
        return run_function(function, X)

cdef np.ndarray run_function(
        node, 
        np.ndarray[np.float32_t, ndim=2] args
        ):
    """
    Runs a function iteratively on the elements contained in the array
    args.
    """
    cdef np.ndarray[np.float32_t, ndim=1] results = np.zeros(args.shape[0], np.float32)
    for i in range(args.shape[0]): 
        results[i] = node.compute_function(args[i,:])

    return results


cdef float compute_fitness(
        functions, 
        np.ndarray[np.float64_t] y_true, 
        np.ndarray[np.float32_t] y_pred, 
        float parsimony_coef, 
        int output_dim
        ):
    """
    Computes fitness for an individual (i.e. functions set). Fitness is
    defined as the sum of mean absolute error and a complexity penalty
    based on the parsimony coefficient passed as a parameter.
    """
    cdef float mae = mean_absolute_error(y_true, y_pred)
    cdef int nodes = 0

    if parsimony_coef != 0:
        for dim in range(output_dim):
            nodes = nodes + functions[f"tree{str(dim)}"].get_length()
        nodes = nodes / output_dim
        return nodes * parsimony_coef + mae
    
    else:
        return mae


cpdef compute_functions_result(
        functions, 
        np.ndarray inputs, 
        np.ndarray y, 
        int output_dim, 
        float min_float, 
        float max_float, 
        float parsimony_coef
        ):
    """
    Computes results and fitness for all functions.
    """
    cdef list performances = []
    cdef np.ndarray y_pred
    cdef np.ndarray[np.float32_t, ndim=1] dim_pred
    cdef float fitness

    for index in range(len(functions)):
        y_pred = np.empty((y.shape[0], y.shape[1]))
        for dim in range(output_dim):
            output_tree = functions.iloc[index][f"tree{str(dim)}"]

            dim_pred = run_function(output_tree, inputs)
            if output_dim == 1:
                y_pred = dim_pred
            else:
                y_pred[:,dim] = dim_pred


        np.nan_to_num(y_pred, copy=False, posinf=max_float, neginf=min_float)
        
        fitness = compute_fitness(
            functions.iloc[index], 
            y, 
            y_pred, 
            parsimony_coef, 
            output_dim
            )
            
        if fitness == np.inf: performances.append(max_float)
        elif fitness == np.NINF: performances.append(min_float)
        elif fitness == np.nan: performances.append(max_float)
        else : performances.append(fitness)

    functions["perf"] = performances
    return functions


    






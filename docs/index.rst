.. symbolic-learn documentation master file, created by
   sphinx-quickstart on Thu Apr 13 17:15:40 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. toctree::
   :maxdepth: 2
   :hidden:

   index
   api_reference

Welcome to symbolic-learn's documentation!
==========================================


symbolic-learn is a sklearn-compatible package that implements a symbolic regression model.


What is symbolic regression?
----------------------------

Symbolic regression is a type of regression model that combines mathematical blocks to find the function that best fits the data. Here each function is represented as a binary tree like this one:

.. image:: _static/genetic_program_tree.png
   :alt: Function tree representation : image not found
   :align: center


The model initially generates a random population of such functions. It then uses genetic programming techniques on it to find out the function that best fits our dataset.
As this model is based on `scikit-learn's <http://scikit-learn.org>`_ base estimator, it can be used the same way you would use any sklearn model. Thus, you can use it in pipelines or apply fine-tuning techniques such as GridSearchCV on it.

Symbolic regression is best used when you want to take a naive approach to solving a regression problem. Unlike most existing models, it does not come with an *a priori* specification of a model. Therefore it is a good idea to use it when you want to find out and understand the mathematical structures in your data. 

Example
---------------------------

Here is how to instantiate and train a symbolic regression model::

   >>> from sblearn.models import SymbolicRegressor
   >>> model = SymbolicRegressor()
   >>> model.fit(X_train, y_train)

After training your model, you can use access the fitted functions' simplified formulas and full function trees through the model's attributes ``formulas`` and ``trees``. Here is an example with a toy dataset where there is a linear relation between the features and our target::

   >>> print(model.formulas)
   ['y0 = 21.227012634277344*x0 + 49.040491104125977']
   >>> print(model.trees[0])
   y0_tree:                                                    
   (*)                                                
   |                                                 
   | ─────── (/)                                     
   |          |                                      
   |          | ─────── x0                           
   |          |                                      
   |          | ─────── x0                           
   |                                                 
   | ─────── (+)                                     
               |                                      
               | ─────── (*)                          
               |          |                           
               |          | ─────── x0                
               |          |                           
               |          | ─────── 21.227012634277344
               |                                      
               | ─────── (+)                          
                        |                           
                        | ─────── 15.254457473754883
                        |                           
                        | ─────── 33.786033630371094

Installation
---------------------------

In order to install the package, use this command::

   pip install symbolic-learn




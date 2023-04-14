"""
Contains the implementation of the classes representing function trees.
The cpdef functions are often intended to act as interfaces between the
pure-Python and Cython parts of the code, and are the only ones callable
from Python.
"""
# Author: Vincent Papelard <papelardvincent@gmail.com>
#
# License: MIT

import cython
import sympy
import numpy as np

cimport numpy as np
from libc.stdlib cimport rand, srand
np.import_array()


cpdef void set_random_state(int random_state):
    srand(random_state)


########################################################
# Available math operations are defined below
########################################################

# Basic math operations are redefined using types in order to
# speed up the code
cdef inline float add(float a, float b):
    return a + b
cdef inline float sub(float a, float b):
    return a - b
cdef inline float mul(float a, float b):
    return a * b
cdef inline float div(float a, float b):
    return a / b


cdef inline float rand_float(float[2] constants_range):
    cdef float diff = constants_range[1] - constants_range[0]
    cdef float random_float = (1.0 / 100000000.0) * (rand() % 100000000.0)

    return constants_range[0] + diff * random_float



cdef char predicate_symb[4]
predicate_symb[0] = '+'
predicate_symb[1] = '-'
predicate_symb[2] = '*'
predicate_symb[3] = '/'

cdef int predicate_arity[4]
predicate_arity[0] = 2
predicate_arity[1] = 2
predicate_arity[2] = 2
predicate_arity[3] = 2

cdef float(*ptr_add)(float, float)
cdef float(*ptr_sub)(float, float)
cdef float(*ptr_mul)(float, float)
cdef float(*ptr_div)(float, float)
ptr_add = add
ptr_sub = sub
ptr_mul = mul
ptr_div = div
ctypedef float (*math_function)(float, float)
cdef math_function predicate_func[4]
predicate_func[0] = ptr_add
predicate_func[1] = ptr_sub
predicate_func[2] = ptr_mul
predicate_func[3] = ptr_div


cdef class Node:
    """
    Base class for all other nodes (functions, constant values and input values).
    Should not be implemented directly.
    """

    cdef Node left # First child
    cdef Node right # Second child

    cdef Node clone(self): 
        """Returns a deep copy of the node. Must be overriden"""
        pass

    
    cpdef float compute_function(
        self, 
        np.ndarray[np.float32_t, ndim=1] input_vector
        ):
        """
        Computes the result of the function applied to the
        input vector and returns a float. Must be overriden.
        """
        pass
    
    cdef Node mutate(
        self, 
        int input_dims, 
        float[2] constants_range
        ): 
        """
        Replaces a random node from the tree. If there is only one node
        in the tree, it is returned as-it-is.
        """

        cdef int choice
        cdef Node new_child
        cdef Node mutated

        if not self.left and not self.right: 
            choice = rand() % 2
            if choice == 0:
                mutated = Input_value(rand() % input_dims)
            else:
                mutated = Constant(rand_float(constants_range))
            return mutated  

        elif self.left and self.right: 
            choice = rand() % 2
            if choice == 0: 
                new_child = self.left.mutate(input_dims, constants_range)
                self.left = new_child
            else: 
                new_child = self.right.mutate(input_dims, constants_range)
                self.right = new_child                
            return self
        
        elif self.left and not self.right:
            new_child = self.left.mutate(input_dims, constants_range)
            self.left = new_child
            return self

    cdef Node adjust(
        self, 
        float coef_range=0.05
        ): 
        """
        Adjusts a node, multiplying it by a random float between 
        1-coef_range and 1+coef_range. Must be overriden.
        """
        pass
    
    cpdef str get_tree(
        self, 
        int lvl=0
        ): 
        """
        A convenience method that returns the full tree as str.
        Must be overriden.
        """
        pass
    
    cpdef int get_length(self):
        """
        Returns the total number of nodes in the tree.
        """

        cdef int length = 0

        if self.left: 
            length += self.left.get_length()
        if self.right: 
            length += self.right.get_length()

        length += 1
        return length

    cpdef str get_expression(self):
        """
        Returns a simplified math formula for the tree. 
        Must be overriden.
        """
        pass
        


cdef class Function(Node):
    """
    Node that represents a function/predicate.
    """

    
    cdef int function_id # ID of the function in the predicate arrays declared
    # at the beginning of this module.

    def __init__(
        self, 
        int function_id
        ):
  
        self.function_id = function_id
        self.left = None
        self.right = None


    cpdef void add_child(
        self, 
        Node child
        ):
        """
        Adds a child to the function node.
        """
        if self.left: 
            self.right = child
        else: 
            self.left = child


    cpdef int get_function_id(self): 
        return self.function_id


    cpdef float compute_function(
        self, 
        np.ndarray[np.float32_t, ndim=1] input_vector
        ):
        """
        Computes the result of the function applied to the
        input vector and returns a float
        """

        if predicate_arity[self.function_id] == 1: # Checks the function arity
            # This will be implemented later in order to support some unary predicates
            # e.g. sin, cos, tan
            raise NotImplemented("Unary predicates support is not implemented yet")
        else:
            return predicate_func[self.function_id](
                self.left.compute_function(input_vector), 
                self.right.compute_function(input_vector)
                )


    cdef Node adjust(
        self, 
        float coef_range=0.05
        ):
        """
        Adjusts the node, multiplying it by a random float between 
        1-coef_range and 1+coef_range.
        """
        cdef float[2] coef_interval = [-coef_range, coef_range]
        if self.function_id == 2:
            if type(self.left) == Constant: 
                self.left.value *= (rand_float(coef_interval) + 1)
                return self
            elif type(self.right) == Constant: 
                self.right.value *= (rand_float(coef_interval) + 1)
                return self

        cdef Function parent = Function(2)
        parent.add_child(self)
        parent.add_child(Constant(rand_float(coef_interval) + 1))
        return parent
        

    cpdef str get_tree(
        self, 
        int lvl=0
        ): 
        """
        A convenience method that prints the full tree (without math
        simplifications).
        """

        cdef list lines = []
        if lvl == 0: 
            lines.append('\n')  
            lines.append(f"({chr(predicate_symb[self.function_id])})\n")
        else: 
            if lvl == 1:
                lines.append(' |\n')
                lines.append(' | ─────── ' + f"({chr(predicate_symb[self.function_id])})\n")
            else:
                lines.append((lvl-1)*' |         ' + ' |\n')
                lines.append((lvl-1)*' |         ' + ' | ─────── ' + f"({chr(predicate_symb[self.function_id])})\n")
        
        if self.left: 
            lines.extend(self.left.get_tree(lvl=lvl+1))
        if self.right: 
            lines.extend(self.right.get_tree(lvl=lvl+1))
        
        cdef str tree_str = ""
        for l in lines: 
            tree_str += l
        if lvl==0: 
            tree_str = self.clean_tree(tree_str)
        return tree_str
    
    @cython.wraparound(True)
    cdef str clean_tree(
        self, 
        str tree_str
        ):
        """
        Cleans a tree passed as list of str to get rid of '|' characters in excess.
        """

        cdef list lines = tree_str.split('\n')
        cdef list grid = []
        cdef bint display_bars = False

        for line in lines:
            characters_list = []
            for character in line: 
                characters_list.append(character)
            grid.append(characters_list)
        
        # Make it so all the rows have the same length (necessary for what comes after):
        cdef int max_row_length = 0
        for row in grid:
            if len(row) > max_row_length:
                max_row_length = len(row)
        for row_index in range(len(grid)):
            for diff in range(max_row_length-len(grid[row_index])): grid[row_index].append(' ')


        for col_index in range(max_row_length):
            display_bars = False

            for row_index in reversed(range(-len(grid)+1, 0)):
                if grid[row_index][col_index] not in (" ", "", '|'):
                    display_bars = False
                elif grid[row_index][col_index] == '|':
                    if len(grid[row_index]) > col_index+2 and \
                    grid[row_index][col_index+1] == ' ' and \
                    grid[row_index][col_index+2] == '─':
                        display_bars = True
                    
                    if not display_bars:
                        grid[row_index][col_index] = ' '

        cdef str str_line = ""
        for row_index in range(len(grid)):
            for column_index in range(len(grid[row_index])):
                str_line += grid[row_index][column_index]
            grid[row_index] = str_line
            str_line = ""

        return '\n'.join(grid)


    cpdef str get_expression(self):
        """
        Returns the function's simplified math formula as a string.
        """

        cdef str expression = chr(predicate_symb[self.function_id])

        if self.left != None: 
            expression = "(" + self.left.get_expression() + ") " + expression
        if self.right != None: 
            expression = expression + " (" + self.right.get_expression() + ")"

        return str(sympy.sympify(expression))
    

    cdef Node clone(self): 
        """
        Returns a deep copy of the whole tree.
        """

        cdef Function cloned_tree = Function(self.function_id)
        if self.left != None: 
            cloned_tree.add_child(self.left.clone())
        if self.right != None: 
            cloned_tree.add_child(self.right.clone())
        return cloned_tree
            
        
cdef class Input_value(Node):
    """
    A node that represents an input value in the formula represented
    by the tree.
    """
    cdef int input_dim # The dimension of the input value in the input vector

    def __init__(
        self, 
        int input_dim
        ):
  
        self.input_dim = input_dim
        self.left = None
        self.right = None


    cpdef float compute_function(
        self, 
        np.ndarray[np.float32_t, ndim=1] input_vector
        ):
        """
        Computes the result of the function applied to the
        input vector and returns a float.
        """
        return input_vector[self.input_dim]


    cdef Node adjust(
        self, 
        float 
        coef_range=0.05
        ):
        """
        Adjusts the node, multiplying it by a random float between 
        1-coef_range and 1+coef_range.
        """
        cdef float[2] coef_interval = [-coef_range, coef_range]
        cdef Function parent = Function(2)
        parent.add_child(self)
        parent.add_child(Constant(rand_float(coef_interval) + 1))
        return parent


    cpdef str get_tree(
        self, 
        int lvl=0
        ): 

        cdef list lines = []
        
        if lvl == 1:
            lines.append(' |\n')
            lines.append(' | ─────── ' + f"x{self.input_dim}\n")
        else:
            lines.append((lvl-1)*' |         ' + ' |\n')
            lines.append((lvl-1)*' |         ' + ' | ─────── ' + f"x{self.input_dim}\n")

        cdef str tree_str = ""
        for l in lines: tree_str += l
        return tree_str


    cpdef str get_expression(self):
        return 'x' + str(self.input_dim)


    cdef Node clone(self): 
        return Input_value(self.input_dim)


cdef class Constant(Node):
    """
    Node that represents a constant value.
    """

    cdef public float value # number stored in the node

    def __init__(
        self, 
        float value
        ):
  
        self.value = value
        self.left = None
        self.right = None
    
    cpdef float compute_function(
        self, 
        np.ndarray[np.float32_t, ndim=1] input_vector
        ):
        
        return self.value
    

    cdef Node adjust(
        self, 
        float coef_range=0.05
        ):

        cdef float[2] coef_interval = [-coef_range, coef_range]
        self.value *= (rand_float(coef_interval) + 1)
        return self


    cpdef str get_tree(
        self, 
        int lvl=0
        ): 

        cdef list lines = []
        if lvl == 1:
            lines.append(' |\n')
            lines.append(' | ─────── ' + f"{self.value}\n")
        else:
            lines.append((lvl-1)*' |         ' + ' |\n')
            lines.append((lvl-1)*' |         ' + ' | ─────── ' + f"{self.value}\n")
        cdef str tree_str = ""
        for l in lines: tree_str += l
        return tree_str


    cpdef str get_expression(self):
        return str(self.value)


    cdef Node clone(self): 
        return Constant(self.value)



cpdef list cross_functions(
    Node parent1, 
    Node parent2
    ):
    """
    Applies crossover to parent1 and parent2, returns
    a list containing the two children generated.
    """
    
    if (parent1.left == None and parent1.right == None) or \
    (parent2.left == None and parent2.right == None):
        return [parent1.clone(), parent2.clone()]
    
    cdef Function root1 = parent1.clone()
    cdef Function root2 = parent2.clone()

    cdef bint p1_change_left
    cdef Node c1
    cdef Node c2

    if root1.right == None: 
        c1 = root1.left
        p1_change_left = True
    else: 
        if rand() % 2 == 0: 
            c1 = root1.left
            p1_change_left = True
        else: 
            c1 = root1.right
            p1_change_left = False

    if root2.right == None: 
        c2 = root2.left
        root2.left = c1

    else: 
        if rand() % 2 == 0: 
            c2 = root2.left
            root2.left = c1
        else: 
            c2 = root2.right
            root2.right = c1

    if p1_change_left: 
        root1.left = c2
    else: root1.right = c2
    
    return [root1, root2]
        

cpdef Function generate_function(
    int max_depth, 
    int input_dim, 
    constants_range
    ):
    """
    Generates a random function tree.
    """

    cdef Function root = Function(rand() % 4) # 4 is the number of predicates currently
    # available, see definitions on top of the file.
    cdef float[2] constants = constants_range

    cdef list current_parents = [root]
    for level in range(max_depth):
        next_parents = []
        for parent in current_parents:
            if not isinstance(parent, Function): 
                continue
            for child in range(predicate_arity[parent.get_function_id()]):
                if level == max_depth-2:
                    if rand() % 2 == 0: 
                        new_node = Constant(rand_float(constants))
                    else: 
                        new_node = Input_value(rand() % input_dim)
                else:
                    if rand() % 2 == 0:
                        if rand() % 2 == 0: 
                            new_node = Constant(rand_float(constants))
                        else: 
                            new_node = Input_value(rand() % input_dim)
                    else: 
                        new_node = Function(rand() % 4)
                parent.add_child(new_node)
                next_parents.append(new_node)
        current_parents = next_parents
                        
    return root


cpdef Node adjust(
    Node current_node, 
    float coef_range
    ):
    """
    Acts as an interface to call the nodes' adjustment functions from
    pure-Python code.
    """
    return current_node.adjust(coef_range)

cpdef Node mutate(
    Node current_node, 
    int input_dim, 
    constants_range
    ):
    """
    Acts as an interface to call the nodes' mutation functions from
    pure-Python code.
    """
    cdef float[2] constants = constants_range
    return current_node.mutate(input_dim, constants)
import numpy as np
from forbiddenfruit import curse
import types

excluded_list = {'arange', 'zeros', 'ones', 'array'}

for numpy_function_name in dir(np):
    potential_function = getattr(np, numpy_function_name)
    if numpy_function_name.lower()==numpy_function_name and hasattr(potential_function, '__call__') \
            and not hasattr(np.ndarray, numpy_function_name) and numpy_function_name not in excluded_list:
        curse(np.ndarray, numpy_function_name, types.MethodType(getattr(np, numpy_function_name), None, np.ndarray))

from numpy import *

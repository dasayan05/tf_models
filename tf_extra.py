"""
Author: Ayan Das
Desc: Some extended functionalities for Tensorflow
"""

from numpy import float_ as float_default_prec

def tf_ph_decor(func):
    def wrapper(names, inps_or_shapes, dtype=float_default_prec):
        if inps_or_shapes is None:
            return func(names, [None]*len(names), dtype)
        else:
            return func(names, inps_or_shapes, dtype)
    return wrapper

@tf_ph_decor
def tf_placeholders(names, inps_or_shapes, dtype=float_default_prec):
    # internal imports
    from tensorflow import placeholder, placeholder_with_default as placeholder_def
    from numpy import ndarray
    
    if not names:
        raise "names shouldn't be None or []"
    if len(names) != len(inps_or_shapes):
        raise "names and inps_or_shapes should have same length"
    
    ph = [] # list of placeholders later to be converted to tuple
    
    for n, name in enumerate(names):
        if type(inps_or_shapes[n]) is tuple or inps_or_shapes[n] is None:
            ph.append( placeholder(dtype=dtype, shape=inps_or_shapes[n], name=name) )
        elif type(inps_or_shapes[n]) is ndarray:
            ph.append( placeholder_def(inps_or_shapes[n], inps_or_shapes[n].shape, name=name) )
        else:
            raise "Must be a list of tuples or ndarrays"
    
    return tuple(ph)

def tf_Variables(names, init_vals, dtype=float_default_prec):
    from tensorflow import Variable

    if (not names) or (not init_vals):
        raise "names and init_vals should be lists"
    if len(names) != len(init_vals):
        raise "length of names and init_vals should be same"

    var = []

    for n, name in enumerate(names):
        var.append( Variable(init_vals[n], name=name, dtype=float_default_prec) )

    return tuple(var)
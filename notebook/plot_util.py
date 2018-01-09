import numpy as np

def get_colormap(y, colors=['r', 'b']):
    return y.replace(dict(zip(np.unique(y), colors)))
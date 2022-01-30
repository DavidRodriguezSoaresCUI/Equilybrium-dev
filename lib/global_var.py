#pylint: disable=no-member
''' Used to store global variables
Singleton pattern used
'''

class DotDict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

GLOBAL_VAR = DotDict()

def get_global_variables() -> dict:
    ''' Equivalent of singleton: get unique instance of `GLOBAL_VAR`
    '''
    return GLOBAL_VAR

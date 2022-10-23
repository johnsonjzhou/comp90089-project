################################################################################
# Wrapper functions for pickle                                                 #
################################################################################

import pickle

def save_pickle(obj, path:str) -> None:
    """
    Saves an object as a pickle
    
    Args:
        obj (obj) An object to be pickled
        path (str) File path, best to have the suffix '.pickle'
    """
    with open(path, "wb") as file:
        pickle.dump(obj, file)
        print(f"Pickle saved {path}")
    return

def load_pickle(path:str) -> object:
    """
    Loads an object from a pickle file
    
    Args:
        path (str) File path to the pickle file
    
    Returns (object)
    """
    with open(path, "rb") as file:
        obj = pickle.load(file)
        print(f"Pickle loaded {path}")
    return obj

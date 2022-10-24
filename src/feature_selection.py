################################################################################
# Helper functions for feature selection                                       #
################################################################################

import numpy as np
from sklearn.decomposition import PCA

def find_n_components(pca, percent_exp_variance):
    """
    Finds the number of components required to reach a desired
    explainable variance
    
    Args:
        pca (PCA): the fitted PCA estimator
        percent_exp_variance (float): perccent of explainable variance required
    
    Returns (int): the number of components required
    """
    # Get cumulative sum of the explained variance ratio list
    cum_ratio = np.cumsum(pca.explained_variance_ratio_)
    # Count the number of elements required to explain percent variance
    n = np.sum([cum_ratio <= (percent_exp_variance / 100)]) + 1
    
    return n

def find_max_contributor(pca, component):
    """
    Finds the max contributor to any component number
    
    Args:
        pca (PCA): the fitted PCA estimator
        component (int): the component number to search
    
    Returns (int): the index of the feature with the highest contribution
    """
    # Get the components from pca estimator, sorted by explained variance
    # Shape (n_components, n_features)
    components = pca.components_
    # Find the index of the largest value at the desired component
    feature_index = np.argmax(components[component])
    
    return feature_index
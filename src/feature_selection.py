################################################################################
# Helper functions for feature selection                                       #
################################################################################

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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

def analyse_pca(df, percent_exp_variance=90):
    """
    Analyse PCA and find the optimal n_components
    
    Args:
        df (DataFrame)
        percent_exp_variance (float): percent of explainable variance required
    """
    # Analyse PCA up to maximum number of features
    max_components = len(df.columns.values)
    pca_analyse = PCA(n_components=max_components, random_state=42)
    pca_analyse = pca_analyse.fit(df)
    
    # Find the number of components required
    n_components = find_n_components(pca_analyse, percent_exp_variance)
    
    # Plot the analysis
    x = range(1, max_components + 1)
    ax = sns.lineplot(x=x, y=np.cumsum(pca_analyse.explained_variance_ratio_))
    ax.set_xticks(x)
    ax.set_xlabel("Number of components")
    ax.set_ylabel("Explained variance ratio (cumulative)")
    plt.axhline(y=(percent_exp_variance / 100), color="red", linestyle="--")
    plt.axvline(x=n_components, color="black", linestyle="-")
    plt.title(
        label=f"PCA analysis: {n_components} required to explain {percent_exp_variance}% variance",
        fontweight="semibold")
    plt.tight_layout()
    plt.show()

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
    feature_index = np.argmax(np.absolute(components[component]))
    
    return feature_index

def list_top_contributors(pca, component, feature_list):
    """
    
    """
    # Get the components from pca estimator, sorted by explained variance
    # Shape (n_components, n_features)
    components = pca.components_
################################################################################
# Helper functions for feature Importantance Analysis
################################################################################
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression

def forestFit(X,y):
    """
    Fit random Forest to the data
    
    Args:
        X: CoS
        y: label of CoS (LoS)
        
    Return:
        feature_names : all feature included
        forest: the fitted forest model
    """
    # split our dataset into training and testing subsets.
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

    feature_names = [f"feature {i}" for i in range(X.shape[1])]
    forest = RandomForestClassifier(random_state=0)
    forest.fit(X_train, y_train)
    return feature_names, forest, X_test, y_test

def LinearRegressionFit(X,y):
    """
    Fit Linear Regression to the data
    
    Args:
        X: CoS
        y: label of CoS (LoS)
        
    Return:
        feature_names : all feature included
        lr: the fitted fLinear Regression model
    """
    # split our dataset into training and testing subsets.
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

    feature_names = [f"feature {i}" for i in range(X.shape[1])]
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    return feature_names, lr, X_test, y_test

def impurityPlot(feature_names,model):
    """
        Feature importance based on mean decrease in impurity
        (maybe use this as a baseline model)
        
        Args:
            feature_names: all feature included
            model: fitted estimator 
            X_test: testing dataset
            y_test: labels of testing data (LoS)
    """
    importances = model.feature_importances_

    # calculate the mean and standard deviation of accumulation of the impurity decrease within each tree
    std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)

    forest_importances = pd.Series(importances, index=feature_names)

    fig, ax = plt.subplots()
    forest_importances.plot.bar(yerr=std, ax=ax)
    ax.set_title("Feature importances using MDI")
    ax.set_ylabel("Mean decrease in impurity")
    fig.tight_layout()

def permutationPlot(feature_names, model, X_test, y_test):
    """
        Feature importance based on feature permutation
        (More costly but less bias toward high-cardinality features by randomly shuffled a single feature value each time) 

        Args:
            feature_names: all feature included
            model: fitted estimator 
            X_test: testing dataset
            y_test: labels of testing data (LoS)
            
    """
    result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=2)
    forest_importances = pd.Series(result.importances_mean, index=feature_names)

    fig, ax = plt.subplots()
    forest_importances.plot.bar(yerr=result.importances_std, ax=ax)
    ax.set_title("Feature importances using permutation on full model")
    ax.set_ylabel("Mean accuracy decrease")
    fig.tight_layout()
    plt.show()
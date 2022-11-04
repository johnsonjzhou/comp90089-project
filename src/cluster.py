################################################################################
# Helper functions for clustering                                              #
################################################################################

import matplotlib.pyplot as plt
from pandas import DataFrame
import matplotlib.cm as cm
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples

from sklearn.neighbors import BallTree
from sklearn.cluster import DBSCAN

def fit_kmeans(args:list) -> tuple[int, KMeans]:
    """
    Fit K-Means to the data
    
    Args(list):
        (int): number of clusters
        (DataFrame): the data
        
    Return(tuple):
        (int): number of clusters
        (KMeans): the fitted KMeans
    """
    # Unpack args
    k, df = args
    
    # Fit K-Means
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(df)
    
    return (k, kmeans)

def analyse_fits(args:list) -> tuple[int, float, float]:
    """
    Use a fitted Kmeans to predict cluster labels,
    then get the SSEw and silhouette_score metrics
    
    Args(list):
        (int): number of clusters
        (KMeans): the fitted KMeans
        (DataFrame): the data
        
    Return(tuple):
        (int): number of clusters
        (float): the within cluster Sum of Squared Error
        (float): the silhouette score
    """
    # Unpack args
    k, fitted_kmeans, df = args

    # The within cluster sum of squared errors is accessible
    # from the inertia_ attribute
    sse_w = fitted_kmeans.inertia_
    
    # Calculate silhouette
    y = fitted_kmeans.predict(df)
    silhouette = silhouette_score(df, y)
    
    return(k, sse_w, silhouette)

def k_means_elbow_analysis(k_list:list, df, title):
    """
    Conducts elbow analysis over a defined list of k and plot the results
    
    Args:
        k_list (list): list of k to examine
        df (DataFrame)
        title (str): the title of the plot
    """
    # Fit K-Means over range of k_list
    kmeans_fits = [fit_kmeans([k, df]) for k in k_list]

    # Analyse K-Means over range of k_list
    kmeans_analysis = [analyse_fits([k, kmeans, df]) for k, kmeans in kmeans_fits]

    # Convert analysis to a DataFrame
    analysis_df = DataFrame(
        kmeans_analysis,
        columns=["k", "sse_w", "silhouette"]
    )

    # Plot the elbow analysis
    fig, ax = plt.subplots()
    ax.set_title(title, fontweight="semibold")
    ax.set_xlabel("k")

    ax.plot(analysis_df["k"], analysis_df["sse_w"], color="blue")
    ax.set_ylabel("SSEw")
    ax.tick_params(axis="y", labelcolor="blue")
    ax.axvspan(xmin=3, xmax=5, ymin=0, ymax=1, alpha=0.2, color="orange")

    ax2 = ax.twinx()
    ax2.plot(analysis_df["k"], analysis_df["silhouette"], color="green")
    ax2.set_ylabel("Average silhouette coefficient")
    ax2.tick_params(axis="y", labelcolor="green")

    plt.show()
    print(analysis_df)
    return

def silhouette_analysis(range_k:list, df, title:str):
    """
    Conduct and plot silhouette analysis over three values of k.
    
    Args:
        range_k (list): A list of three values of k to compare.
        df (DataFrame)
        title (str): The title to display for the plot

    Adapted from:
    https://scikit-learn.org/0.24/auto_examples/cluster/plot_kmeans_silhouette_analysis.html
    """

    # Create a subplot with 1 row and 3 columns
    fig, axs = plt.subplots(1, 3)
    fig.set_size_inches(10, 7)

    for i, k in enumerate(range_k):
        # The silhouette coefficient can range from -1, 1 but in this example all
        # lie within [-0.1, 1]
        axs[i].set_xlim([-0.1, 1])
        # The (k+1)*10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly.
        axs[i].set_ylim([0, len(df) + (k + 1) * 10])

        # Initialize the clusterer with k value and a random generator
        # seed of 10 for reproducibility.
        clusterer = KMeans(n_clusters=k, random_state=42)
        cluster_labels = clusterer.fit_predict(df)

        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        silhouette_avg = silhouette_score(df, cluster_labels)
        print("For k =", k,
            "The average silhouette_score is :", silhouette_avg)

        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(df, cluster_labels)

        y_lower = 10
        for j in range(k):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            jth_cluster_silhouette_values = \
                sample_silhouette_values[cluster_labels == j]

            jth_cluster_silhouette_values.sort()

            size_cluster_j = jth_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_j

            color = cm.nipy_spectral(float(j) / k)
            axs[i].fill_betweenx(np.arange(y_lower, y_upper),
                            0, jth_cluster_silhouette_values,
                            facecolor=color, edgecolor=color, alpha=0.7)

            # Label the silhouette plots with their cluster numbers at the middle
            axs[i].text(-0.05, y_lower + 0.5 * size_cluster_j, str(j))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        axs[i].set_title(f"k={k}")
        axs[i].set_xlabel("Silhouette coefficient")
        axs[i].set_ylabel("Cluster label")

        # The vertical line for average silhouette score of all the values
        axs[i].axvline(x=silhouette_avg, color="red", linestyle="--")

        axs[i].set_yticks([])  # Clear the yaxis labels / ticks
        axs[i].set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])


        plt.suptitle(title, fontweight="semibold")

    plt.show()
    return

def calculate_k_distances(args):
    """
    Calculate k-distances using euclidean distance with the BallTree algorithm
    
    Args(tuple):
        k (int)
        df (DataFrame)
    
    Returns(array): k-distances in descending order of shape n_instances
    """
    # Unpack args
    k, df = args
    
    # BallTree is efficient algorithm defaults to euclidean distance
    # Generate the distance to the k-th nearest neighbour
    tree = BallTree(df)
    distances, _ = tree.query(df, k=k)
    
    # Sort the distances in descending order
    k_distances = np.sort(distances[:,-1])[::-1]
    
    return k_distances

def dbscan_kdist_analysis(args:tuple):
    """
    Explores k-distances for dbscan across a range of k values and plots
    the results on a k-distances graph
    
    Args(tuple)
        k_list (list of int): values of k to explore
        df (DataFrame)
    """
    # Unpack the args
    k_list, df = args
    
    # Create the plot
    plt.style.use("default")
    fig, ax = plt.subplots()

    for k in k_list:
        # Calculate the k_distances
        k_dist = calculate_k_distances(args=(k, df))

        # Add to the plot
        ax.plot(df.index, k_dist, label=f"k={k}")
        
    plt.title(f"K-Distances")
    plt.xlabel("Instances")
    plt.ylabel("k-distance")
    plt.legend()
    plt.show()

def dbscan_kdist_analysis_zoom(args:tuple):
    """
    Explores k-distances for dbscan across a range of k values and plots
    the results on a k-distances graph
    
    Args(tuple)
        k_list (list of int): values of k to explore
        df (DataFrame)
        xlim (int): x-scale limit
        ylim (int): y-scale limit
        title (str): title for the plot
    """
    # Unpack the args
    k_list, df, xlim, ylim, title = args
    
    # Create the plot
    plt.style.use("default")
    fig, ax = plt.subplots()
    ax.set_xlim(right=xlim)
    ax.set_ylim(top=ylim)

    for k in k_list:
        # Calculate the k_distances
        k_dist = calculate_k_distances(args=(k, df))

        # Add to the plot
        ax.plot(df.index, k_dist, label=f"k={k}")
        
    plt.title(title, fontweight="bold")
    plt.xlabel("Instances")
    plt.ylabel("k-distance")
    plt.legend()
    plt.show()

def assign_dbscan_multidensity(args:tuple):
    """
    Uses DBSCAN to assign clusters across a range of eps distances to
    handle multiple densities. Clusters are assigned iteratively from most dense
    to least dense and each run only works with points that are unassigned or
    considered as outlier from the previous run.
    
    Args(tuple):
        eps_list (list): a list of eps distances to use
        min_pts (int): the min_samples parameter of DBSCAN
        df (DataFrame)
    
    Returns(DataFrame): of cluster assignments
    """
    # Unpack the args
    eps_list, min_pts, df = args
    
    df["cluster"] = np.nan
    
    for i, eps in enumerate(np.sort(eps_list)):
        # Initiate DBSCAN
        dbscan = DBSCAN(eps=eps, min_samples=min_pts, n_jobs=-1)
        
        # Select only the instances where cluster is NA or -1 (outlier)
        df_run = df[(df["cluster"].isna() | (df["cluster"] < 0))]
        
        # Skip if no more rows to apply for this run
        if df_run.shape[0] < 1:
            continue
        print(f"DBSCAN run:{i} eps:{eps} n:{df_run.shape[0]}")
        
        # Run DBSCAN to fit the labels
        x = df_run.drop(columns=["cluster"])
        df_run["dbscan"] = dbscan.fit_predict(x)
        
        # Apply a modifier to the labels to distinguish between runs
        # run 1: 1xxx, run 2: 2xxx, etc..
        df_run["dbscan"] = df_run["dbscan"] \
            .apply(lambda x: (x, x + (1000 * (i + 1)))[x >= 0])
        
        # Join the results to the original DataFrame
        df_run = df_run[["dbscan"]]
        df = pd.merge(left=df, right=df_run, how="left", on="ID")
        
        # Assign cluster based on dbscan assignments
        df.loc[df["dbscan"].notna(), "cluster"] = df["dbscan"]
        
        # Clean up after run
        df = df.drop(columns=["dbscan"])
    
    assigned_clusters = df["cluster"]
    return assigned_clusters

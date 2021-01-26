import csv
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.plotting import parallel_coordinates
from pyspark.mllib.clustering import KMeans
from sklearn.manifold import TSNE


def calculate_wssse(data_to_cluster):
    """Calculates Within Set Sum of Squared Error (WSSSE)"""
    
    K = []
    wssse = []
    
    for k in range(2, 12):
        
        print('Computing WSSE for {}'.format(k))
        K.append(k)
        model = KMeans.train(
            data_to_cluster, k, maxIterations=10, initializationMode='random')
        wssse_value = model.computeCost(data_to_cluster)
        wssse.append(wssse_value)
     
    # Plot the WSSSE for different values of k
    plt.plot(K, wssse)
    plt.show()
    
    
def get_cluster_ids(data_to_cluster, K):
    """Gets cluster ids for data to be clustered"""
    
    #~ print('Traning KMeans')
    model = KMeans.train(
        data_to_cluster, K, maxIterations=10, initializationMode='random')
    #~ print('Finished Traning')
    #~ print('Predicting Cluster IDs')
    cluster_ids = model.predict(data_to_cluster)
    #~ print('Finished Prediction')
    #~ print('10 Sample Cluster IDs:')
    #~ print(cluster_ids.takeSample(False, 10))
    #~ print('10 Sample Data (nutrient-nutrient) Clusters')
    #~ print(data_to_cluster.takeSample(False, 10))
    #~ print('Fetched Cluster IDs')
    
    return cluster_ids, data_to_cluster
    
    
def plot_clusters(cluster_ids, clusters, K):
    """Reduces dimensions of data to plots clusters on a 2d plane"""

    plt.figure(1)
        
    # Cluster ID values are used to show clusters with color in plots
    colors = cluster_ids.collect()
    normalize = mpl.colors.Normalize(vmin=0, vmax=K)
    
    # Reduce nutrient values to 2 dimensions with TSNE
    nutrient_values = clusters.collect()
    cluster_embedded = TSNE(n_components=2).fit_transform(nutrient_values)
    
    # Plot clusters
    content_on_x = cluster_embedded[:,0]
    content_on_y = cluster_embedded[:,1]
    plt.scatter(
        content_on_x, content_on_y, s=3, c=colors, cmap="jet", norm=normalize)
    plt.show()
    plt.clf()


def categorize_values(categorize_value):
    """Classifies data into financial classes"""
    
    compare_value = float(categorize_value[0])
    if (compare_value<2):
        categorize_value[0]='lower'
    
    if (compare_value>=2 and compare_value<4):
        categorize_value[0]='middle'

    if (compare_value>=4):
        categorize_value[0]='upper'
        
    return categorize_value


def plot_as_4d(relevant_data, year):
    """Plots parallel coordinates"""
    
    categorized_data = relevant_data.map(categorize_values)
    #~ print(categorized_data.takeSample(False, 5))
    csv_data = categorized_data.collect()

    fieldnames = ['Class', 'Carbohydrates', 'Fiber', 'Fat', 'Protein']

    with open('./{}_categorized_data.csv'.format(year), 'w') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames = fieldnames)
        writer.writeheader()
        wr = csv.writer(csv_file)
        wr.writerows(csv_data)
    
    data = pd.read_csv('./{}_categorized_data.csv'.format(year))
    plt.figure()
    parallel_coordinates(data, 'Class', color=['#1f77b4','#2ca02c','#d62728'])
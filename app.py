import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MeanShift, estimate_bandwidth, KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import skfuzzy as fuzz
import os
import pickle
import matplotlib.pyplot as plt


# Load and sample data
df_cleaned_modelling = pd.read_csv("homicide_modelling.csv")
sampled_df = df_cleaned_modelling.sample(frac=0.1, random_state=42)
sampled_df2 = df_cleaned_modelling.sample(frac=0.05, random_state=42)
numerical_data = sampled_df[['month_encoded', 'year_normalized']]
numerical_data2 = sampled_df2[['month_encoded', 'year_normalized']]

# Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(numerical_data.values)

def load_or_train_model(algo_name, data, num_clusters, use_pca):
    model_filename = f"PickleFile/{algo_name}PCA-{num_clusters}.pkl" if use_pca else f"PickleFile/{algo_name}-{num_clusters}.pkl"

    if os.path.exists(model_filename):
        with open(model_filename, 'rb') as f:
            return pickle.load(f)
    
    model = {'MeanShiftClustering': MeanShift(bandwidth=estimate_bandwidth(data, quantile=0.2, n_samples=500)),
             'KMeansClustering': KMeans(n_clusters=num_clusters, random_state=42),
             'AgglomerativeClustering': AgglomerativeClustering(n_clusters=num_clusters)}[algo_name]
    model.fit(data)

    with open(model_filename, 'wb') as f:
        pickle.dump(model, f)
    return model

# Clustering
def perform_clustering(algorithm, data, num_clusters, use_pca=False):
    if algorithm == 'FuzzyCMeansClustering':
        centers, u, _, _, _, _, fpc = fuzz.cluster.cmeans(
            data.T, num_clusters, 2.0, error=0.0001, maxiter=100, init=None)
        labels = np.argmax(u, axis=0)
        st.write("Fuzzy Partition Coefficient (FPC): " + str(fpc))
        return labels, centers
    else:
        model = load_or_train_model(algorithm, data, num_clusters, use_pca)
        if algorithm == 'MeanShiftClustering':
            return model.labels_, model.cluster_centers_
        elif algorithm == 'KMeansClustering':
            if use_pca:
                pca = PCA(n_components=2)
                data_pca = pca.fit_transform(data)
                return model.labels_, data_pca
            st.write("Inertia value: " + str(model.inertia_))
            return model.labels_, model.cluster_centers_
        else:
            st.write("Silhouette Score: " + str(silhouette_score(numerical_data2, model.labels_)))
            return model.labels_

# Main function
def main():
    st.title('Clustering Visualisation App')

    # Choose clustering algorithm, number of clusters (if applicable), and whether to use PCA (for K-Means only)
    clustering_algorithm = st.sidebar.selectbox('Choose Clustering Algorithm', ['Mean Shift', 'Fuzzy C-means', 'K-means', 'Agglomerative'])
    num_clusters = st.sidebar.slider('Number of Clusters', min_value=2, max_value=10, value=3) if clustering_algorithm != 'Mean Shift' else None
    use_pca = st.sidebar.checkbox('Use PCA', value=False) if clustering_algorithm == 'K-means' else False

    # Perform clustering based on user selection
    if clustering_algorithm == 'Mean Shift':
        labels, cluster_centers = perform_clustering("MeanShiftClustering", numerical_data, num_clusters)
    elif clustering_algorithm == 'Fuzzy C-means':
        labels, cluster_centers = perform_clustering("FuzzyCMeansClustering", numerical_data, num_clusters)
    elif clustering_algorithm == 'K-means':
        if use_pca:
            labels, data_pca = perform_clustering("KMeansClustering", X_scaled, num_clusters, use_pca)
        else:
            labels, cluster_centers = perform_clustering("KMeansClustering", numerical_data, num_clusters)
    else:
        labels = perform_clustering("AgglomerativeClustering", numerical_data, num_clusters)
        sampled_df2['cluster_label'] = labels

    # Plot the clustered data
    plt.figure(figsize=(8, 6))

    plot_data = data_pca if use_pca else numerical_data.values
    for label in np.unique(labels):
        if clustering_algorithm == "Agglomerative":
            cluster_data = sampled_df2[sampled_df2['cluster_label'] == label]
            plt.scatter(cluster_data['month_encoded'], cluster_data['year_normalized'], label=f'Cluster {label}')
        else:
            cluster_data = plot_data[labels == label]
            plt.scatter(cluster_data[:, 0], cluster_data[:, 1], label=f'Cluster {label}')
    if clustering_algorithm in ['Fuzzy C-means', 'K-means', 'Mean Shift'] and use_pca is False:
        plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], s=100, c='black')
    
    plt.title('Clustering Result')
    plt.xlabel('Principal Component 1' if use_pca else 'Month_Encoded')
    plt.ylabel('Principal Component 2' if use_pca else 'Year_Normalized')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.grid(True)
    st.pyplot()

if __name__ == '__main__':
    st.set_option('deprecation.showPyplotGlobalUse', False)
    main()

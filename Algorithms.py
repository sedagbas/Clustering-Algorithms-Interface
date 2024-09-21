# -*- coding: utf-8 -*-
"""
Created on Sat Jun  8 18:01:25 2024

@author: sedat
"""
from PyQt5 import QtWidgets
from ParameterDialogs import *
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering, MeanShift, estimate_bandwidth, AffinityPropagation, DBSCAN, KMeans
from scipy.cluster.hierarchy import linkage, dendrogram
from Actions import Actions
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import QVBoxLayout, QProgressBar
import time
import sip
from collections import deque

""" 
@ brief Algorithms class, derived from Actions class 
"""
class Algorithms(Actions):
    def __init__(self, coor={}, undoInitial=deque(), redoInitial=deque(), undoFinal=deque(), redoFinal=deque(), clusters={}, centers=np.array([]), closestPoints={}, figure=None):
        super().__init__(coor)
        self.__figure = figure
        self.__clusters = clusters
        self.__centers = centers
        self.__closestPoints = closestPoints
        
        
    """ 
    @ brief Set function for clusters 
    @ param labels Clusters' labels
    """
    def setClusters(self, labels):
        self.__clusters = labels
        
    """
    @ brief Set function for center of the clusters  
    @ param centers Centers of the clusters
    """
    def setClusterCenters(self, centers):
        centers_shape = centers.shape
        self.__centers = np.zeros(centers_shape)
        self.__centers = centers.copy()
        
    """ 
    \brief Set function for closest points 
    \param points Closest points to the cluster centers
    """
    def setClosestPoints(self, points):
        self.__closestPoints = points
    
    """ 
    \brief Get function for clusters 
    \return Clusters
    """
    def getClusters(self):
        return self.__clusters

    """ 
    \brief Get function for center of clusters 
    \return Centers of the clusters
    """
    def getClusterCenters(self):
        return self.__centers
    
    """ 
    \brief Get function for closest points 
    \return Closest points to the cluster centers
    """
    def getClosestPoints(self):
        return self.__closestPoints
        
    """ 
    \brief Set function for the figure
    \param figure Figure to be set
    """
    def setFigure(self, figure):
        self.__figure = figure
        
    """ 
    \brief Get function for the figure 
    \return The figure
    """
    def getFigure(self):
        return self.__figure
        
    """ 
    \brief Draw clustering results on the canvas 
    """
    def drawClustering(self):
        canvas = FigureCanvas(self.getFigure())
        layout = QVBoxLayout()
    
        progress_bar = QProgressBar(self.finalGraph)
        progress_bar.setGeometry(0, self.finalGraph.height() - 20, self.finalGraph.width(), 20)
        progress_bar.setObjectName("progress_bar")
        progress_bar.setRange(0, 100)
        progress_bar.setValue(0)
        progress_bar.show()            

        self.statusbar.showMessage("Loading")
        
        for i in range(101):
            progress_bar.setValue(i)
            time.sleep(0.005) 
            
        self.statusbar.clearMessage()
        progress_bar.deleteLater()
    
        if self.finalGraph.layout() is not None:
            old_layout = self.finalGraph.layout()
            while old_layout.count():
                child = old_layout.takeAt(0)
                if child.widget():
                    child.widget().deleteLater()
            sip.delete(old_layout)

        layout.addWidget(canvas)
        self.finalGraph.setLayout(layout)
        
        ax = self.getFigure().axes[0]
        for i, (x, y) in enumerate(self.getCoordinates().values()):
            ax.scatter(x, y)

        """ Color palette """
        colors = plt.cm.Spectral(np.linspace(0, 1, len(self.getClusters())))
        
        """ Show points using different colors for each cluster  """
        for label, cluster_points in self.getClusters().items():
            color = colors[label % len(colors)]  
            cluster_points = np.array([point[0] for point in cluster_points])
            ax.scatter(cluster_points[:, 0], cluster_points[:, 1], c=color, label=f'Cluster {label}')
            for i, (x, y) in enumerate(cluster_points):
                ax.text(x, y, str(i), fontsize=9, ha='right', va='bottom')  
        
        self.writeToInfoPanel()   
        
    """ 
    \brief K-Means clustering method 
    """
    def k_Means(self): 
        """ Open a parameter dialog for K-Means """
        param_dialog = KMeansParamDialog(self)
        if param_dialog.exec_() == QDialog.Accepted:
            params = param_dialog.get_params()
            n_clusters = params['n_clusters']
            init = params['init']
            max_iter = params['max_iter']
            algorithm = params['algorithm']

            """ Convert coordinates dictionary to numpy array """
            coordinates_array = np.array(list(self.getCoordinates().values()))

            """ Create K-Means model and fit it with the data """
            kmeans = KMeans(n_clusters=n_clusters, init=init, max_iter=max_iter, algorithm=algorithm)
            kmeans.fit(coordinates_array)
            labels = kmeans.labels_
            cluster_centers = kmeans.cluster_centers_

            """ Organize labels into a dictionary """
            clustered_dict = self.organizeClusters(labels, coordinates_array)
            
            closest_points = self.find_closest_points_to_centers(cluster_centers, clustered_dict)

            """ Plot the results """
            fig, ax = plt.subplots()
            self.setClusters(clustered_dict)
            self.setClusterCenters(cluster_centers.copy())
            self.setClosestPoints(closest_points)
            
            self.__enableFinal__()
            self.setFigure(fig)
            self.__setUndoInitial__(fig)
            self.plot_clustering_result(ax, "K-Means Clustering")
                        
    """ 
    \brief Affinity Propagation clustering method 
    """
    def affinityPropagationClustering(self):  
        """ Open a parameter dialog for Affinity Propagation """
        param_dialog = AffinityPropagationParamDialog(self)
        if param_dialog.exec_() == QDialog.Accepted:
            params = param_dialog.get_params()
            damping = params['damping']
            preference = params['preference']

            coordinates_dict = self.getCoordinates()
            
            """ Convert dictionary to array """
            coordinates_array = np.array(list(coordinates_dict.values()))

            """ Create Affinity Propagation model """
            affinity_model = AffinityPropagation(damping=damping, preference=preference)

            """ Fit the Affinity Propagation model """
            affinity_model.fit(coordinates_array)

            """ Get cluster centers and labels """
            cluster_centers = affinity_model.cluster_centers_
            labels = affinity_model.labels_
            n_clusters = len(cluster_centers)
            
            """ Organize clusters """
            clusters = self.organizeClusters(labels, coordinates_array)

            """ Find the index of the point closest to the center in each cluster """
            closest_points = self.find_closest_points_to_centers(cluster_centers, clusters)

            """ Draw the output """
            fig, ax = plt.subplots(figsize=(8, 6))
            self.setClusters(clusters)
            self.setClusterCenters(cluster_centers)
            self.setClosestPoints(closest_points)
            
            self.__enableFinal__()
            self.setFigure(fig)
            self.__setUndoInitial__(fig)
            self.plot_clustering_result(ax, "Affinity Propagation Clustering")
            
            
    """
    \brief Mean Shift clustering method
    """
    def meanShiftClustering(self):
        """
        \brief Open a parameter dialog for Mean Shift
        """
        param_dialog = MeanShiftParamDialog(self)
        if param_dialog.exec_() == QDialog.Accepted:
            params = param_dialog.get_params()
            quantile = params['quantile']
            n_samples = params['n_samples']

            """
            \brief Estimate the bandwidth
            """
            coordinates_dict = self.getCoordinates()
            coordinates_array = np.array(list(coordinates_dict.values()))
            bandwidth = estimate_bandwidth(coordinates_array, quantile=quantile, n_samples=n_samples)

            """
            \brief Create Mean Shift model
            """
            ms_model = MeanShift(bandwidth=bandwidth, bin_seeding=True)

            """
            \brief Fit the Mean Shift model
            """
            ms_model.fit(coordinates_array)

            """
            \brief Get labels and cluster centers
            """
            labels = ms_model.labels_
            cluster_centers = ms_model.cluster_centers_
            n_clusters = len(np.unique(labels))

            """
            \brief Organize clusters
            """
            clusters = self.organizeClusters(labels, coordinates_array)
            
            closest_points = self.find_closest_points_to_centers(cluster_centers, clusters)

            """
            \brief Draw the output
            """
            fig, ax = plt.subplots(figsize=(8, 6))
            self.setClusters(clusters)
            self.setClusterCenters(cluster_centers.copy())
            self.setClosestPoints(closest_points)
            
            self.__enableFinal__()
            self.setFigure(fig)
            self.__setUndoInitial__(fig)
            self.plot_clustering_result(ax, "Mean-Shift Clustering")
            

    """
    \brief Spectral clustering method
    """
    def spectralClusteringMethod(self):
        """
        \brief Open a parameter dialog for Spectral Clustering
        """
        param_dialog = SpectralParamDialog(self)
        if param_dialog.exec_() == QDialog.Accepted:
            params = param_dialog.get_params()
            n_clusters = params['n_clusters']
            affinity = params['affinity']
            assign_labels = params['assign_labels']
            n_neighbors = params['n_neighbors']
            
            """
            \brief Convert the coordinate dictionary to a numpy array
            """
            coordinates_dict = self.getCoordinates()
            coordinates_array = np.array(list(coordinates_dict.values()))
            
            spectral_model = SpectralClustering(
                n_clusters=n_clusters, 
                affinity=affinity, 
                assign_labels=assign_labels,
                n_neighbors=n_neighbors
            )

            labels = spectral_model.fit_predict(coordinates_array)

            """
            \brief Organize clusters
            """
            clusters = self.organizeClusters(labels, coordinates_array)
            
            cluster_centers = []
            for label, points in clusters.items():
                center = np.mean(np.array([point[0] for point in points]), axis=0)
                cluster_centers.append(center)
                
            closest_points = self.find_closest_points_to_centers(cluster_centers, clusters)
                
            """
            \brief Set up the figure and plot
            """
            fig, ax = plt.subplots(figsize=(8, 6))
            colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
            for label in np.unique(labels):
                cluster_points = np.array([coord for coord, idx in clusters[label]])
                ax.scatter(cluster_points[:, 0], cluster_points[:, 1], c=colors[label % len(colors)], label=f'Cluster {label}')
                for i, (x, y) in enumerate(cluster_points):
                    ax.text(x, y, str(i), fontsize=9, ha='right', va='bottom')
                    
                    """
                    \brief Plot cluster centers
                    """
                for center in cluster_centers:
                    ax.scatter(center[0], center[1], c='black', marker='x')
        
            """
            \brief Set cluster information and draw
            """
            self.setClusters(clusters)
            self.setClusterCenters(np.array(cluster_centers)) 
            self.setClosestPoints(closest_points)
        
            self.__enableFinal__()
            self.setFigure(fig)
            self.__setUndoInitial__(fig)
            self.plot_clustering_result(ax, "Spectral Clustering")


    """
    \brief Hierarchical clustering method
    """
    def hierarchical_Clustering(self):
    
        """
        \brief Open a parameter dialog for Hierarchical Clustering
        """
        param_dialog = HierarchicalParamDialog(self)
        if param_dialog.exec_() == QDialog.Accepted:
            params = param_dialog.get_params()
            linkage_method = params['linkage_method']
            
            """
            \brief Apply hierarchical clustering method
            """
            coordinates_dict = self.getCoordinates()
            coordinates_array = np.array(list(coordinates_dict.values()))
            Z = linkage(coordinates_array, method=linkage_method)
            
            """
            \brief Draw the dendrogram
            """
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.set_title('Hierarchical Clustering Dendrogram')
            ax.set_xlabel('Sample Index')
            ax.set_ylabel('Distance')
            dendrogram(Z, ax=ax)
            
            self.__enableFinal__()
            self.setFigure(fig)
            self.__setUndoInitial__(fig)
            self.drawClustering()
            
    """
    \brief DBSCAN clustering method
    """
    def dbscanClustering(self):
        """
        \brief Open a parameter dialog for DBSCAN Clustering
        """
        param_dialog = DBSCANParamDialog(self)
        if param_dialog.exec_() == QDialog.Accepted:
            params = param_dialog.get_params()
            eps = params['eps']
            min_samples = params['min_samples']


        """
        \brief Create the DBSCAN model
        """
        dbscan_model = DBSCAN(eps=eps, min_samples=min_samples)

        """
        \brief Apply the DBSCAN model on the data
        """
        coordinates_dict = self.getCoordinates()
        coordinates_array = np.array(list(coordinates_dict.values()))
        labels = dbscan_model.fit_predict(coordinates_array)
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels) - (1 if -1 in labels else 0)
        
        """
        \brief Set up clusters
        """
        clusters = {}
        for i, label in enumerate(labels):
            if label != -1:
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append(coordinates_array[i])

        """
        \brief Visualize the clustering results
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        colors = plt.cm.Spectral(np.linspace(0, 1, n_clusters))
        for label, color in zip(unique_labels, colors):
            if label != -1:
                cluster_label = f'Küme {label}'
                if label in clusters:
                    cluster_points = np.array(clusters[label])
                    ax.scatter(cluster_points[:, 0], cluster_points[:, 1], color=color, label=cluster_label)
                    for i, (x, y) in enumerate(cluster_points):
                        ax.text(x, y, str(i), fontsize=9, ha='right', va='bottom')
                        
        self.__enableFinal__()
        self.setFigure(fig)
        self.__setUndoInitial__(fig)
        self.plot_clustering_result(ax, "DBSCAN Clustering")
    

    def plot_clustering_result(self, ax, title):
        clusters = self.getClusters()
        cluster_centers = self.getClusterCenters()
        colors = plt.cm.Spectral(np.linspace(0, 1, len(clusters))) 
        
        for idx, (cluster_id, points) in enumerate(clusters.items()):
            points = np.array([point[0] for point in points])
            ax.scatter(points[:, 0], points[:, 1], label=f'Cluster {idx + 1}', color=colors[idx % len(colors)])   

        ax.scatter(cluster_centers[:, 0], cluster_centers[:, 1], marker='x', s=60, color='k', label='Centers')

        ax.set_title(title)
        ax.legend()

        self.drawClustering()
    
    """
    \brief Find closest points to cluster centers
    """
    def find_closest_points_to_centers(self, cluster_centers, clustered_dict):
        closest_points = {}
        for label, center in enumerate(cluster_centers):
            cluster_points = np.array([point[0] for point in clustered_dict[label]])
            distances = np.linalg.norm(cluster_points - center, axis=1)
            closest_index = np.argmin(distances)
            closest_points[label] = clustered_dict[label][closest_index][1]
        
    """
    \brief Organize clusters
    """
    def organizeClusters(self, labels, coordinates_array):
        clustered_dict = {}
        for i, label in enumerate(labels):
            if label not in clustered_dict:
                clustered_dict[label] = []
            clustered_dict[label].append((tuple(coordinates_array[i]), i))
        return clustered_dict
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  8 18:25:24 2024

@author: sedat
"""

from PyQt5.QtWidgets import QDialog, QFormLayout, QLineEdit, QComboBox, QPushButton

class KMeansParamDialog(QDialog):
    """
    Dialog for setting K-Means clustering parameters.
    """
    def __init__(self, parent=None):
        """
        Initializes the KMeansParamDialog.

        @param parent Parent widget.
        """
        super().__init__(parent)
        self.setWindowTitle('K-Means Parameters')
        self.layout = QFormLayout(self)

        self.n_input = QLineEdit(self)
        self.n_input.setText('8')  
        self.layout.addRow('n:', self.n_input)

        self.init_input = QComboBox(self)
        self.init_input.addItems(['k-means++', 'random'])
        self.layout.addRow('init:', self.init_input)

        self.max_iter_input = QLineEdit(self)
        self.max_iter_input.setText('300')
        self.layout.addRow('max_iter:', self.max_iter_input)

        self.algorithm_input = QComboBox(self)
        self.algorithm_input.addItems(['lloyd', 'elkan'])
        self.layout.addRow('algorithm:', self.algorithm_input)

        self.submit_btn = QPushButton('Submit', self)
        self.submit_btn.clicked.connect(self.accept)
        self.layout.addWidget(self.submit_btn)

    def get_params(self):
        """
        Returns the parameters set by the user.

        @return A dictionary with the K-Means parameters.
        """
        return {
            'n_clusters': int(self.n_input.text()),
            'init': self.init_input.currentText(),
            'max_iter': int(self.max_iter_input.text()),
            'algorithm': self.algorithm_input.currentText()
        }

class SpectralParamDialog(QDialog):
    """
    Dialog for setting Spectral Clustering parameters.
    """
    def __init__(self, parent=None):
        """
        Initializes the SpectralParamDialog.

        @param parent Parent widget.
        """
        super().__init__(parent)
        self.setWindowTitle('Spectral Clustering Parameters')
        self.layout = QFormLayout(self)

        self.n_clusters_input = QLineEdit(self)
        self.n_clusters_input.setText('2')  
        self.layout.addRow('n_clusters:', self.n_clusters_input)

        self.affinity_input = QComboBox(self)
        self.affinity_input.addItems(['nearest_neighbors', 'rbf'])
        self.layout.addRow('affinity:', self.affinity_input)

        self.assign_labels_input = QComboBox(self)
        self.assign_labels_input.addItems(['kmeans', 'discretize'])
        self.layout.addRow('assign_labels:', self.assign_labels_input)

        self.n_neighbors_input = QLineEdit(self)
        self.n_neighbors_input.setText('10') 
        self.layout.addRow('n_neighbors:', self.n_neighbors_input)

        self.submit_btn = QPushButton('Submit', self)
        self.submit_btn.clicked.connect(self.accept)
        self.layout.addWidget(self.submit_btn)

    def get_params(self):
        """
        Returns the parameters set by the user.

        @return A dictionary with the Spectral Clustering parameters.
        """
        return {
            'n_clusters': int(self.n_clusters_input.text()),
            'affinity': self.affinity_input.currentText(),
            'assign_labels': self.assign_labels_input.currentText(),
            'n_neighbors': int(self.n_neighbors_input.text())
        }

class HierarchicalParamDialog(QDialog):
    """
    Dialog for setting Hierarchical Clustering parameters.
    """
    def __init__(self, parent=None):
        """
        Initializes the HierarchicalParamDialog.

        @param parent Parent widget.
        """
        super().__init__(parent)
        self.setWindowTitle('Hierarchical Clustering Parameters')
        self.layout = QFormLayout(self)

        self.linkage_method_input = QComboBox(self)
        self.linkage_method_input.addItems(['ward', 'single', 'complete', 'average', 'centroid', 'median'])
        self.layout.addRow('Linkage Method:', self.linkage_method_input)

        self.submit_btn = QPushButton('Submit', self)
        self.submit_btn.clicked.connect(self.accept)
        self.layout.addWidget(self.submit_btn)

    def get_params(self):
        """
        Returns the parameters set by the user.

        @return A dictionary with the Hierarchical Clustering parameters.
        """
        return {
            'linkage_method': self.linkage_method_input.currentText()
        }

class MeanShiftParamDialog(QDialog):
    """
    Dialog for setting Mean Shift Clustering parameters.
    """
    def __init__(self, parent=None):
        """
        Initializes the MeanShiftParamDialog.

        @param parent Parent widget.
        """
        super().__init__(parent)
        self.setWindowTitle('Mean Shift Clustering Parameters')
        self.layout = QFormLayout(self)

        self.quantile_input = QLineEdit(self)
        self.quantile_input.setText('0.2') 
        self.layout.addRow('Quantile:', self.quantile_input)

        self.n_samples_input = QLineEdit(self)
        self.n_samples_input.setText('500')
        self.layout.addRow('Number of Samples:', self.n_samples_input)

        self.submit_btn = QPushButton('Submit', self)
        self.submit_btn.clicked.connect(self.accept)
        self.layout.addWidget(self.submit_btn)

    def get_params(self):
        """
        Returns the parameters set by the user.

        @return A dictionary with the Mean Shift Clustering parameters.
        """
        return {
            'quantile': float(self.quantile_input.text()),
            'n_samples': int(self.n_samples_input.text())
        }

class DBSCANParamDialog(QDialog):
    """
    Dialog for setting DBSCAN Clustering parameters.
    """
    def __init__(self, parent=None):
        """
        Initializes the DBSCANParamDialog.

        @param parent Parent widget.
        """
        super().__init__(parent)
        self.setWindowTitle('DBSCAN Clustering Parameters')
        self.layout = QFormLayout(self)

        self.eps_input = QLineEdit(self)
        self.eps_input.setText('0.5') 
        self.layout.addRow('Epsilon (eps):', self.eps_input)

        self.min_samples_input = QLineEdit(self)
        self.min_samples_input.setText('5') 
        self.layout.addRow('Minimum Samples:', self.min_samples_input)

        self.submit_btn = QPushButton('Submit', self)
        self.submit_btn.clicked.connect(self.accept)
        self.layout.addWidget(self.submit_btn)

    def get_params(self):
        """
        Returns the parameters set by the user.

        @return A dictionary with the DBSCAN Clustering parameters.
        """
        return {
            'eps': float(self.eps_input.text()),
            'min_samples': int(self.min_samples_input.text())
        }

class AffinityPropagationParamDialog(QDialog):
    """
    Dialog for setting Affinity Propagation Clustering parameters.
    """
    def __init__(self, parent=None):
        """
        Initializes the AffinityPropagationParamDialog.

        @param parent Parent widget.
        """
        super().__init__(parent)
        self.setWindowTitle('Affinity Propagation Clustering Parameters')
        self.layout = QFormLayout(self)

        self.damping_input = QLineEdit(self)
        self.damping_input.setText('0.5')  
        self.layout.addRow('Damping:', self.damping_input)

        self.preference_input = QLineEdit(self)
        self.preference_input.setText('None')  
        self.layout.addRow('Preference:', self.preference_input)

        self.submit_btn = QPushButton('Submit', self)
        self.submit_btn.clicked.connect(self.accept)
        self.layout.addWidget(self.submit_btn)

    def get_params(self):
        """
        Returns the parameters set by the user.

        @return A dictionary with the Affinity Propagation Clustering parameters.
        """
        return {
            'damping': float(self.damping_input.text()),
            'preference': None if self.preference_input.text() == 'None' else float(self.preference_input.text())
        }

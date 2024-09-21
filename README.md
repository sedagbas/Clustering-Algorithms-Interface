# Clustering-Algorithms-Interface
This project aims to design a user-friendly interface for clustering data, leveraging the capabilities of the scikit-learn library. Clustering is an essential technique in data analysis and machine learning that groups similar data points together, revealing hidden patterns and insights.

  1. The “Ui_MainWindow” class defines the characteristics and behaviors of a user interface created using PyQt5. This class contains helper methods to access and manage specific features of the interface. These methods are used to enable or disable certain functionalities related to clustering processes.

  2. The "Actions" class is derived from the Ui_MainWindow class in PyQt5, which creates a graphical user interface (GUI). The project includes various methods for performing graphical operations. Specifically, this class contains functions for users to input and save coordinates, visualize these coordinates on a graph, export these graphs, and write the results of certain operations to an information panel. Additionally, there are methods for tasks such as reading coordinates from a file, saving coordinates to a file, and exporting graphs. 

  3. The “Algorithms” class is derived from the “Actions” class. This class contains methods to apply various clustering algorithms such as K-Means, Affinity Propagation, Mean Shift, Spectral Clustering, Hierarchical Clustering, and DBSCAN. It opens appropriate parameter dialogs to ask the user for the necessary parameters for each algorithm. Then, it calculates the results of the selected algorithm, visualizes them, and presents these results to the user.

  4. The "Parameter Dialogs" module allows users to adjust the parameters of various clustering algorithms via a graphical user interface using PyQt5, and enables running these algorithms. It provides a dialog box for each parameter adjustment, collecting the user's input parameters, and then executes the relevant clustering algorithm.

OUTPUTS 

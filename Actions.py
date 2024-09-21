# -*- coding: utf-8 -*-
"""
Created on Sat Jun  8 16:33:21 2024

@author: sedat
"""

from project import Ui_MainWindow
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QMainWindow, QFileDialog, QVBoxLayout, QMessageBox, QApplication
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import sip
from PyQt5.QtWidgets import QProgressBar
import time
import matplotlib.pyplot as plt
import numpy as np
from collections import deque

"""
\class Actions
\brief Actions class, derived from QMainWindow and Ui_MainWindow.
"""
class Actions(QMainWindow, Ui_MainWindow):
    """
    \brief Constructor for Actions class.
    \param coor Dictionary of coordinates.
    """
    def __init__(self, coor={}, undoInitial=deque(), redoInitial=deque(), undoFinal=deque(), redoFinal=deque()):
        super().__init__()
        self.__coordinates = coor
        self.__undoInitial = undoInitial
        self.__redoInitial = redoInitial
        self.__undoFinal = undoFinal
        self.__redoFinal = redoFinal
        
        self.setupUi(self)
        
    """
    \brief Set function for coordinates.
    \param index Index of the coordinate.
    \param x X-coordinate.
    \param y Y-coordinate.
    """
    def setCoordinates(self, index, x, y):
        self.__coordinates[index] = (x,y)
    
    """
    \brief Get function for coordinates.
    \return Dictionary of coordinates.
    """
    def getCoordinates(self):
        return self.__coordinates
    
    """ 
    @ brief Set function for undoInitial 
    @ param layout Layout to be set for undoInitial
    """
    def __setUndoInitial__(self, layout):
        self.__undoInitial.append(layout)
        
    """ 
    @ brief Set function for redoInitial 
    @ param layout Layout to be set for redoInitial
    """
    def __setRedoInitial__(self, layout):
        self.__redoInitial.append(layout)
    
    """ 
    @ brief Set function for undoFinal 
    @ param layout Layout to be set for undoFinal
    """    
    def __setUndoFinal__(self, layout):
        self.__undoFinal.append(layout)
      
    """ 
    @ brief Set function for redoFinal 
    @ param layout Layout to be set for redoFinal   
    """
    def __setRedoFinal__(self, layout):
        self.__redoFinal.append(layout)
        
    """ 
    @ brief Get function for undoInitial 
    @ return The undoInitial deque
    """
    def getUndoInitial(self):
        return self.__undoInitial
    
    """ 
    @ brief Get function for redoInitial 
    @ return The redoInitial deque
    """
    def getRedoInitial(self):
        return self.__redoInitial
    
    """ 
    @ brief Get function for undoFinal 
    @ return The undoFinal deque
    """
    def getUndoFinal(self):
        return self.__undoFinal
    
    """ 
    @ brief Get function for redoFinal 
    @ return The redoFinal deque
    """
    def getRedoFinal(self):
        return self.__redoFinal
    
    """
    \brief Method to clear initial or final solutions based on sender.
    """
    def clear(self):
        sender = self.sender()
        if sender == self.clearInitial or sender == self.actionClearInitialSolution:
            self.__clearInitialSolution__()
            self.statusbar.showMessage("Initial part is cleaned", 2000)
        elif sender == self.clearFinal or sender == self.actionClearFinalSolution:
            self.__clearFinalSolution__()
            self.statusbar.showMessage("Final part is cleaned", 2000)

    """
    \brief Method to clear initial solution data.
    """
    def __clearInitialSolution__(self):
        self.__coordinates.clear()
        self.textEditInfoPanel.clear()
        
        if self.initialGraph.layout() is not None:
            old_layout = self.initialGraph.layout()
            while old_layout.count():
                child = old_layout.takeAt(0)
                if child.widget():
                    child.widget().deleteLater()
            sip.delete(old_layout)
                    
        self.__disableInitial__()
          
    """
    \brief Method to clear final solution data.
    """
    def __clearFinalSolution__(self):   
        if self.finalGraph.layout() is not None:
            old_layout = self.finalGraph.layout()
            while old_layout.count():
                child = old_layout.takeAt(0)
                if child.widget():
                    child.widget().deleteLater()
            sip.delete(old_layout)
                    
        self.__disableFinal__()
        
    """
    \brief Method to save coordinates.
    """
    def saveCoordinates(self):
        sender = self.sender()
        if sender == self.saveInitial or sender == self.actionSaveInitialSolution:
            self.__saveInitialCoordinates__()
            self.statusbar.showMessage("Initial coordinates are saved.", 2000)
        elif sender == self.saveFinal or sender == self.actionSaveFinalSolution:
            self.__saveFinalCoordinates__()
            self.statusbar.showMessage("Final coordinates are saved.", 2000)
        
    """
    \brief Method to save initial solution coordinates to a file.
    """
    def __saveInitialCoordinates__(self):
        filename, _ = QFileDialog.getSaveFileName(self, 'Save Initial Coordinates', '', 'Text Files (*.txt)')
        if filename:
            with open(filename, 'w') as file:
                for index, (x, y) in self.getCoordinates().items():
                    file.write(f"{index} {x} {y}\n")
            
    """
    \brief Method to save final solution coordinates to a file.
    """
    def __saveFinalCoordinates__(self):
        clus = self.getClusters()
        if clus:
            file_dialog = QFileDialog(self)
            file_dialog.setAcceptMode(QFileDialog.AcceptSave)
            file_dialog.setWindowTitle("Save Final Coordinates")
            file_dialog.setFileMode(QFileDialog.AnyFile)
            file_dialog.setNameFilter("Text Files (*.txt);;All Files (*)")

            if file_dialog.exec_():
                file_path = file_dialog.selectedFiles()[0]
                with open(file_path, 'w') as file:
                    for cluster_label, coordinates in clus.items():
                        file.write(f"Cluster {cluster_label}:\n")
                        for x, y in coordinates:
                            file.write(f"{x}, {y}\n")  
                        file.write("\n")  

    """
    \brief Method to export graphs based on sender.
    """
    def exportGraph(self):
        sender = self.sender()
        if sender == self.exportAsInitial or sender == self.actionExportInitialSolution:
            self.__exportInitialGraph__()
            self.statusbar.showMessage("Initial graph is exported", 2000)
        elif sender == self.exportAsFinal or sender == self.actionExportFinalSolution:
            self.__exportFinalGraph__()
            self.statusbar.showMessage("Final graph is exported", 2000)        
        
    """
    \brief Method to export initial graph as an image.
    """
    def __exportInitialGraph__(self):
        filename, _ = QFileDialog.getSaveFileName(self, 'Export Initial Graph', '', 'Image Files (*.jpg)')
        if filename:
            canvas = self.initialGraph.findChild(FigureCanvas)
            if canvas:
                fig = canvas.figure
                fig.savefig(filename, format='jpg')
            else:
                QMessageBox.critical(self, "Error", "No initial graph to save.")
          
    """
    \brief Method to export final graph as an image.
    """
    def __exportFinalGraph__(self):
        if self.finalGraph.layout() is not None:
            file_path, _ = QFileDialog.getSaveFileName(self, "Save Final Graph", "", "JPEG Files (*.jpg);;All Files (*)")
            if file_path:
                pixmap = self.finalGraph.grab()
                pixmap.save(file_path, "JPEG")
        else:
            QMessageBox.critical(self, "Error", "No final graph to save.")
    
    """
    \brief Method to handle close event.
    \param event Close event.
    """
    def closeEvent(self, event):
        reply = QMessageBox.question(self, 'Exit Confirmation', 
                                     "Are you sure you want to exit?", 
                                     QMessageBox.Yes | QMessageBox.No, 
                                     QMessageBox.No)
        if reply == QMessageBox.Yes:
            QApplication.quit()
        else:
            event.ignore()
            
    """
    \brief Method to open a file and read coordinates.
    """
    def openFile(self):
        filename, _ = QFileDialog.getOpenFileName(self, 'Open File', '.', 'Text Files (*.txt)')
        if filename:
            self.__clearInitialSolution__()
        
            with open(filename, 'r') as file:
                lines = file.readlines()
                index = 1
                for line in lines:
                    """
                    \brief Split the line by spaces to get x and y coordinates.
                    """
                    coordinates = line.strip().split()
                    if len(coordinates) == 2:
                        x, y = coordinates
                        self.setCoordinates(index, float(x), float(y))
                        index += 1
                    else:
                        QMessageBox.critical(self, "Error", "Ignore invalid line: {line.strip()}")
                    
            self.__enableInitial__()
            self.__drawInitialGraph__()
            self.writeToInfoPanel()
                
    """
    \brief Method to draw the initial graph.
    """
    def __drawInitialGraph__(self):
        if self.getCoordinates():
            progress_bar = QProgressBar(self.initialGraph)
            progress_bar.setGeometry(0, self.initialGraph.height() - 20, self.initialGraph.width(), 20)
            progress_bar.setObjectName("progress_bar")
            progress_bar.setRange(0, 100)
            progress_bar.setValue(0) 
            progress_bar.show()            
            
            self.statusbar.showMessage("Loading...")  
            
            for i in range(101):
                progress_bar.setValue(i)
                time.sleep(0.005)  
            self.statusbar.clearMessage()

            """
            \brief Remove the progress bar once complete.
            """
            progress_bar.deleteLater()
            
            fig, ax = plt.subplots()
            for index, (x, y) in self.getCoordinates().items():
                ax.plot(x, y, 'ko')
                ax.text(x, y, str(index), fontsize=9, ha='right', va='bottom')

            ax.set_xlabel('X')
            ax.set_ylabel('Y')

            fig.tight_layout()

            """
            \brief Create a canvas to embed the matplotlib figure into PyQt5.
            """
            canvas = FigureCanvas(fig)

            """
            \brief Clear the QWidget and add the canvas.
            """
            layout = QVBoxLayout()
            if self.initialGraph.layout() is not None:
                old_layout = self.initialGraph.layout()
                while old_layout.count():
                    child = old_layout.takeAt(0)
                    if child.widget():
                        child.widget().deleteLater()
                sip.delete(old_layout)
            layout.addWidget(canvas)
            self.initialGraph.setLayout(layout)
                        
       
    def writeToInfoPanel(self):
        """
        @brief Writes cluster information to the info panel.

        This method retrieves clusters, cluster centers, and the closest points to the cluster centers,
        and writes this information to the info panel in a formatted manner.
        """
        clusters = self.getClusters()
        cluster_centers = self.getClusterCenters()  
        closest_points = self.getClosestPoints() 

        # Initialize info_text
        info_text = ""

        if cluster_centers is not None:
            info_text += "Cluster Centers:\n"
            for center in cluster_centers:
                info_text += f"{center[0]}, {center[1]}\n"
            info_text += "\n"

            if clusters:
                for cluster_label, coordinates in clusters.items():
                    info_text += f"Cluster {cluster_label}:\n"
                    for (x, y), _ in coordinates:  
                        info_text += f"{x}, {y}\n"
                    info_text += "\n"  
            else:
                info_text += "Clusters are not set yet.\n"

            if closest_points:
                info_text += "Cluster Center Nodes:\n"
                for label, index in closest_points.items():
                    point = self.coordinates[list(self.coordinates.keys())[index]]
                    info_text += f"Cluster {label}: Point Index {index} -> {point}\n"
                info_text += "\n"
        else:
            info_text += "Cluster centers are not set yet.\n"

        self.textEditInfoPanel.setPlainText(info_text)
        
    def __drawFinalGraph__(self, solution):
        if self.getCoordinates():
            progress_bar = QProgressBar(self.finalGraph)
            progress_bar.setGeometry(0, self.finalGraph.height() - 20, self.finalGraph.width(), 20)
            progress_bar.setObjectName("progress_bar")
            progress_bar.setRange(0, 100)
            progress_bar.setValue(0)
            progress_bar.show()            
            
            self.statusbar.showMessage("Loading...")  
            
            for i in range(101):
                progress_bar.setValue(i)
                time.sleep(0.005)

            self.statusbar.clearMessage()
            progress_bar.deleteLater()

            fig, ax = plt.subplots()
            coordinates = self.getCoordinates()
            clusters = self.getClusters()

            colors = plt.cm.Spectral(np.linspace(0, 1, len(clusters)))

            for index, (x, y) in coordinates.items():
                ax.plot(x, y, 'ko')  # Noktaları siyah renkte çiz
                ax.text(x, y, str(index), fontsize=9, ha='right', va='bottom')

            for cluster_idx, cluster_points in clusters.items():
                cluster_color = colors[cluster_idx % len(colors)]
                cluster_points = np.array([point[0] for point in cluster_points])
                ax.plot(cluster_points[:, 0], cluster_points[:, 1], 'o', color=cluster_color, label=f'Cluster {cluster_idx}')
                
            ax.plot(solution[:, 0], solution[:, 1], 'ro')  

            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_title(self.getTitle())
            ax.legend()

            fig.tight_layout()

            canvas = FigureCanvas(fig)
            layout = QVBoxLayout()
            if self.finalGraph.layout() is not None:
                old_layout = self.finalGraph.layout()
                while old_layout.count():
                    child = old_layout.takeAt(0)
                    if child.widget():
                        child.widget().deleteLater()
                sip.delete(old_layout)
            layout.addWidget(canvas)
            self.__setUndoFinal__(canvas)
            self.finalGraph.setLayout(layout)
            
            self.__enableFinal__()
            
            
    def undo_Final(self):
        """
        @brief Performs undo action for the output.
        """
        self.actionRedoFinal.setEnabled(True)
        self.redoFinal.setEnabled(True)
    
        self.__setRedoFinal__(self.getUndoFinal().pop())
        status_message = "Undone action for output"
        
        if self.getUndoFinal():
            layout = QVBoxLayout()
            layout.addWidget(self.getUndoFinal()[-1])
            
            self.finalGraph.setLayout(layout)
        
        else:
            for widget in self.finalGraph.findChildren(QtWidgets.QWidget):
                widget.deleteLater()
            self.actionUndoFinal.setEnabled(False)
            self.undoFinal.setEnabled(False)
            status_message += " (No more undo available)"
        self.statusbar.showMessage(status_message, 1000)

    def redo_Final(self):
        """ Performs redo action for the output. """
        last = self.getRedoFinal().pop()
        self.__drawFinalGraph__(last)
        self.__setUndoFinal__(last)
        status_message = "Redone action for output"
        
        if not self.getRedoFinal():
            self.actionRedoFinal.setEnabled(False)
            self.redoFinal.setEnabled(False)
            status_message += " (No more redo available)"
            
        self.statusbar.showMessage(status_message, 1000)
        
    
    """
    \brief Performs undo action for the output
    """
    def undo_Initial(self):
        self.actionRedoInitial.setEnabled(True)
        self.redoInitial.setEnabled(True)
        
        self.__setRedoInitial__(self.getUndoInitial().pop())
        status_message = "Undone action for output"
    
        if self.getUndoInitial():
            self.setFigure(self.getUndoInitial()[-1])
            self.drawClustering()
    
        else:
            for widget in self.initialGraph.findChildren(QtWidgets.QWidget):
                widget.deleteLater()
                self.actionUndoInitial.setEnabled(False)
                self.undoInitial.setEnabled(False)
        self.__disableInitial__()
        status_message += " (No more undo available)"
        self.statusbar.showMessage(status_message, 1000)

    """
    \brief Performs redo action for the output
    """
    def redo_Initial(self):
        self.__enableInitial__()
        last = self.getRedoInitial().pop()
        self.setFigure(last)
        self.drawClustering()
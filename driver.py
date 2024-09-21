# -*- coding: utf-8 -*-
"""
Created on Sat Jun  8 16:32:04 2024

@author: sedat
"""

import sys
from PyQt5.QtWidgets import QMainWindow, QAction, qApp, QApplication, QMenu
from Algorithms import Algorithms

class MyWindow(QMainWindow):
    def __init__(self):
        super(MyWindow, self).__init__()
        self.ui = Algorithms()
        self.ui.setupUi(self)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MyWindow()
    ex.show()
    sys.exit(app.exec_())
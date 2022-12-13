#############################################################################
##
## Copyright (C) 2022 nuclei.io
## Contact: https://www.nuclei.io/
##
## This script author: Zhi Huang
## This script contributors: Zhi Huang
## This script created on: 07/25/2022
#############################################################################


from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QAction, QBrush, QColor, QConicalGradient,
    QCursor, QFont, QFontDatabase, QGradient,
    QIcon, QImage, QKeySequence, QLinearGradient,
    QPainter, QPalette, QPixmap, QRadialGradient,
    QTransform)
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWidgets import (QApplication, QHBoxLayout, QVBoxLayout, QGridLayout, QStackedLayout, QMainWindow, QMenu, QLabel,
    QMenuBar, QPlainTextEdit, QSizePolicy, QSplitter, QScrollBar, QProgressBar, QPushButton, QDial, QToolBar,
    QStatusBar, QWidget)


class StatusBar(QWidget):

    def __init__(self,
                MainWindowObject):      
        super(StatusBar, self).__init__()
        self.MainWindow = MainWindowObject
        self.initUI()




    def initUI(self):

        self.statusbar = QStatusBar(self.MainWindow)
        self.statusbar.setContentsMargins(0,0,0,0)
        self.message = QLabel("All good.")
        #self.message.setObjectName("message")
        self.message.setStyleSheet('border: 0; color: black')
        self.message.setMinimumWidth(400)
        self.statusbar_pbar_label = QLabel("Working progress:")
        self.statusbar_pbar = QProgressBar()
        COMPLETED_STYLE = """
                                QProgressBar {
                                border: 1px solid grey;
                                background-color: #FFFFFF;
                                border-radius: 5px;
                                height: 10px;
                                text-align: center;
                                color: #000000;
                                }

                                QProgressBar::chunk {
                                background-color: #009122;
                                width: 3px;
                                border-radius: 2px;
                                margin: 0px;
                                }
                                """
        self.statusbar_pbar.setStyleSheet(COMPLETED_STYLE)
        self.statusbar_pbar.setRange(0, 100)
        self.statusbar_pbar.setFixedWidth(200)
        self.statusbar_pbar.setAlignment(Qt.AlignCenter)
        self.statusbar_pbar.setValue(0)

        ed = QPushButton('Hide toolset')

        #self.statusbar.reformat()
        #self.statusbar.setStyleSheet('border: 0; background-color: #FFF8DC;')
        self.statusbar.setStyleSheet("QStatusBar::item {border: none;}") 
        
        #self.statusbar.addPermanentWidget(QtWidgets.QFrame.VLine())    # <---
        self.statusbar.addWidget(self.message)
        #self.statusbar.addPermanentWidget(QtWidgets.QFrame.VLine())    # <---
        self.statusbar.addPermanentWidget(self.statusbar_pbar_label)
        self.statusbar.addPermanentWidget(self.statusbar_pbar)
        
        #self.statusbar.addPermanentWidget(QtWidgets.QFrame.VLine())    # <---
        self.statusbar.addPermanentWidget(ed)
        #self.statusbar.addPermanentWidget(QtWidgets.QFrame.VLine())    # <---
        self.statusbar.setObjectName("statusbar")
        
        self.statusbar.setFixedHeight(35)
        self.statusbar.setStyleSheet("background-color: white")
        self.statusbar.setContentsMargins(0, 0, 0, 0)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0,0,0,0)
        layout.addWidget(self.statusbar)
        return layout
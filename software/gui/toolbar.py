#############################################################################
##
## Copyright (C) 2022 nuclei.io
## Contact: https://www.nuclei.io/
##
## This script author: Zhi Huang
## This script contributors: Zhi Huang
## This script created on: 07/30/2022
#############################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QGuiApplication, QAction, QBrush, QColor, QConicalGradient,
    QCursor, QFont, QFontDatabase, QGradient,
    QIcon, QImage, QKeySequence, QLinearGradient,
    QPainter, QPalette, QPixmap, QRadialGradient,
    QTransform)

from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWidgets import (QApplication, QHBoxLayout, QVBoxLayout, QGridLayout, QStackedLayout, QMainWindow, QMenu, QLabel,
    QMenuBar, QPlainTextEdit, QSizePolicy, QSplitter, QScrollBar, QProgressBar, QPushButton, QDial, QToolBar, QMessageBox,
    QStatusBar, QWidget)

import os
from functools import partial


class ToolBar(QWidget):

    def __init__(self,
                MainWindowObject,
                iconwidth=20,
                iconheight=20):      
        super(ToolBar, self).__init__()
        self.MainWindow = MainWindowObject
        self.iconwidth = iconwidth
        self.iconheight = iconheight
        self.initUI()



    def initUI(self):

        wd = self.MainWindow.wd
        self.toolBar = QToolBar()
        
        self.icon = {}
        self.icon['View'] = QAction(QIcon(os.path.join(wd, 'Artwork', 'toolbar', "cursor-icon.png")),"View",self.MainWindow)
        self.icon['Ruler'] = QAction(QIcon(os.path.join(wd, 'Artwork', 'toolbar', "ruler-measurement-icon.png")),"Ruler",self.MainWindow)
        self.icon['FreeRuler'] = QAction(QIcon(os.path.join(wd, 'Artwork', 'toolbar', "measuring-tape-icon.png")),"Free ruler",self.MainWindow)
        self.icon['Ellipse'] = QAction(QIcon(os.path.join(wd, 'Artwork', 'toolbar', "ellipse-shape-line-icon.png")),"Circle",self.MainWindow)
        self.icon['Rect'] = QAction(QIcon(os.path.join(wd, 'Artwork', 'toolbar', "object-select-icon.png")),"Rectangle",self.MainWindow)
        self.icon['Polygon'] = QAction(QIcon(os.path.join(wd, 'Artwork', 'toolbar', "hexagon-shape-icon.png")),"Polygon",self.MainWindow)
        self.icon['Freehand'] = QAction(QIcon(os.path.join(wd, 'Artwork', 'toolbar', "signature-icon.png")),"Freehand",self.MainWindow)
        self.icon['Eraser'] = QAction(QIcon(os.path.join(wd, 'Artwork', 'toolbar', "eraser-icon.png")),"Eraser",self.MainWindow)
        self.icon['Text'] = QAction(QIcon(os.path.join(wd, 'Artwork', 'toolbar', "font-icon.png")),"Text",self.MainWindow)
        self.icon['MagicWand'] = QAction(QIcon(os.path.join(wd, 'Artwork', 'toolbar', "magic-icon.png")),"Magic wand",self.MainWindow)


        self.icon['Screenshot'] = QAction(QIcon(os.path.join(wd, 'Artwork', 'toolbar', "camera-icon.png")),"Screenshot",self.MainWindow)
        self.icon['Fullscreen'] = QAction(QIcon(os.path.join(wd, 'Artwork', 'toolbar', "full-screen-arrow-icon.png")),"Full screen",self.MainWindow)
        self.icon['Discussion'] = QAction(QIcon(os.path.join(wd, 'Artwork', 'toolbar', "chat-icon.png")),"Discussion",self.MainWindow)
        self.icon['Twitter'] = QAction(QIcon(os.path.join(wd, 'Artwork', 'toolbar', "twitter-app-icon.png")),"Twitter",self.MainWindow)
        self.icon['Upload'] = QAction(QIcon(os.path.join(wd, 'Artwork', 'toolbar', "cloud-upload-icon.png")),"Upload to cloud",self.MainWindow)
        self.icon['Download'] = QAction(QIcon(os.path.join(wd, 'Artwork', 'toolbar', "cloud-download-icon.png")),"Download to local",self.MainWindow)
        self.icon['Live'] = QAction(QIcon(os.path.join(wd, 'Artwork', 'toolbar', "live-icon.png")),"Live",self.MainWindow)


        for mode in self.icon:
            if mode in ['View','Ellipse','Rect','Polygon']:
                self.icon[mode].setCheckable(True)
            else:
                self.icon[mode].setCheckable(False)
            self.toolBar.addAction(self.icon[mode])
            self.icon[mode].triggered.connect(partial(self.setUIMode, mode))


        self.icon['View'].setChecked(True)
        self.toolBar.setIconSize(QSize(self.iconwidth, self.iconheight))
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0,0,0,0)
        layout.addWidget(self.toolBar)
        return layout


    def setUIMode(self,
                    mode: str):
        if (self.MainWindow.ui.mode == 'Polygon') and not (mode == 'Polygon'): #and (self.MainWindow.ui.annotationMode>0):
            reply = QMessageBox.question(self.MainWindow,
                                        'Question',
                                        'Do you want to stop your polygon annotation? Hint: If you want to move the image during annotation, hold shift while dragging the image.', QMessageBox.Yes, QMessageBox.No)

            if reply == QMessageBox.Yes:
                self.MainWindow.ui.mode = mode
            else:
                self.MainWindow.ui.mode = 'Polygon'
        else:
            self.MainWindow.ui.mode = mode
        
        if self.MainWindow.ui.mode in ['Ruler', 'FreeRuler', 'Ellipse', 'Rect', 'Polygon', 'Freehand', 'Screenshot']:
            self.MainWindow.setCursor(Qt.CrossCursor)
        elif self.MainWindow.ui.mode in ['Text']:
            self.MainWindow.setCursor(Qt.IBeamCursor)
        elif self.MainWindow.ui.mode in ['MagicWand']:
            self.MainWindow.setCursor(Qt.PointingHandCursor)
        else:
            self.MainWindow.setCursor(Qt.ArrowCursor)


        # reset all other icon button.
        for m in self.icon:
            if self.MainWindow.ui.mode != m: self.icon[m].setChecked(False)

        return
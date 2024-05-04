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
from PySide6.QtWidgets import (QApplication, QHBoxLayout, QGridLayout, QStackedLayout, QMainWindow, QMenu, QLabel,
    QMenuBar, QPlainTextEdit, QSizePolicy, QSplitter, QScrollBar, QProgressBar, QPushButton, QDial, QToolBar,
    QStatusBar, QWidget)
from gui.zoomSlider import ZoomSlider
from gui.dial_knob import ValueDial
from gui.toolbar import ToolBar
from gui.statusbar import StatusBar
from gui.toggleButtons import ToggleButtonPanel

from functools import partial


class Ui_MainWindow(object):
    def setupUI(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(1600, 900)
        MainWindow.setUnifiedTitleAndToolBarOnMac(True)
    
        self.mode = 'View' # This is the initial mode when start the program. For more information, check out toolbar.py.
        self.polygon_in_progress = False # status of drawing a polygon



        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")

        self.gridLayout = QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName("gridLayout")
        self.gridLayout_margin = 5
        self.gridLayout.setContentsMargins(self.gridLayout_margin,
                                            self.gridLayout_margin,
                                            self.gridLayout_margin,
                                            self.gridLayout_margin) # left, top, right, bottom margins.

        self.horizontalSplitter = QSplitter(self.centralwidget)
        self.horizontalSplitter.splitterMoved.connect(MainWindow.onSplitterMoved)
        self.horizontalSplitter.setObjectName(u"splitter")
        self.horizontalSplitter.setHandleWidth(2)
        self.horizontalSplitter.setOrientation(Qt.Horizontal)
        self.horizontalSplitter.setStyleSheet("QSplitter::handle:vertical {{margin: 0px 0px; "
                           "background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0, "
                           "stop:0.0 rgba(255, 255, 255, 0),"
                           "stop:0.5 rgba({0}, {1}, {2}, {3}),"
                           "stop:0.99 rgba(255, 255, 255, 0))}}"
                           "QSplitter::handle:horizontal {{margin: 0px 0px; "
                           "background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1, "
                           "stop:0.0 rgba(255, 255, 255, 0),"
                           "stop:0.5 rgba({0}, {1}, {2}, {3}),"
                           "stop:0.99 rgba(255, 255, 255, 0));}}".format(64, 143, 255, 255))

        #self.horizontalLayout.setContentsMargins(0,0,0,0)
        self.horizontalSplitter.setObjectName("horizontalLayout")
        



        self.mainImageGridLayout = QGridLayout(self.centralwidget)
        self.mainImageGridLayout.setObjectName("verticalLayout")
        self.mainImageGridLayout.setContentsMargins(0, 0, 0, 0) # left, top, right, bottom margins.
        self.mainImageGridLayout.setSpacing(0)

        self.MainCanvas_layout = QStackedLayout()
        self.MainCanvas_layout.setStackingMode(QStackedLayout.StackAll)

        self.MainImage = QLabel()
        sizePolicy = QSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.MainImage.sizePolicy().hasHeightForWidth())
        self.MainImage.setSizePolicy(sizePolicy)
        self.MainImage.setContentsMargins(0, 0, 0, 0)
        
        self.MainImage.setMinimumSize(QSize(400, 400))
        self.MainImage.setLineWidth(0)
        self.MainImage.setAlignment(Qt.AlignCenter)
        self.MainImage.setObjectName("MainImage")
        self.MainImage.setStyleSheet('background-color: rgb(200,200,200);')
        
        self.MainCanvas_layout.addWidget(self.MainImage)


        self.MachineLearningOverlay = QLabel()
        sizePolicy = QSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.MachineLearningOverlay.sizePolicy().hasHeightForWidth())
        self.MachineLearningOverlay.setSizePolicy(sizePolicy)
        self.MachineLearningOverlay.setContentsMargins(0, 0, 0, 0)
        self.MainCanvas_layout.addWidget(self.MachineLearningOverlay)

        self.AnnotationOverlay = QLabel()
        sizePolicy = QSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.AnnotationOverlay.sizePolicy().hasHeightForWidth())
        self.AnnotationOverlay.setSizePolicy(sizePolicy)
        self.AnnotationOverlay.setContentsMargins(0, 0, 0, 0)
        self.MainCanvas_layout.addWidget(self.AnnotationOverlay)

        self.DrawingOverlay = QLabel()
        sizePolicy = QSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.DrawingOverlay.sizePolicy().hasHeightForWidth())
        self.DrawingOverlay.setSizePolicy(sizePolicy)
        self.DrawingOverlay.setContentsMargins(0, 0, 0, 0)
        self.MainCanvas_layout.addWidget(self.DrawingOverlay)

        '''
        In MainCanvas layout, we have four overlay:
        Level 1: MainImage (the WSI)
        Level 2: Machine learning Overlay (show machine learning predictions, including nuclei/ROI segmenation)
        Level 3: Annotation Overlay (show all the annotations)
        Level 4: Drawing Overlay (only show the active drawing events)
        
        Just like a building, level 4 is on top of level 3, level 2, and level 1.
        '''
        
        self.rotation_dial = ValueDial(minimum=0, maximum=24, singleStep=1, wrapping=True) # 360/15 = 24
        self.rotation_dial.setFixedWidth(120)
        self.rotation_dial.setFixedHeight(120)
        self.rotation_dial.setNotchesVisible(True)

        self.CanvasUtility_parentWidget = QWidget()
        self.CanvasUtility_layout = QGridLayout()
        self.CanvasUtility_layout.setContentsMargins(0, 0, 0, 0)
        self.CanvasUtility_parentWidget.setLayout(self.CanvasUtility_layout)


        self.OverviewLabel = QLabel()
        self.OverviewLabel.setFixedWidth(300)
        self.OverviewLabel.setFixedHeight(200)
        self.OverviewLabel.setStyleSheet('''background-color: rgb(240,240,240); border:solid black;
                                        border-width : 1px 1px 1px 1px;'''
                                        )
        self.placeholder_1 = QLabel()
        
        self.zoomSlider = ZoomSlider(MainWindow)
        sizePolicy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.zoomSlider.sizePolicy().hasHeightForWidth())
        self.zoomSlider.setSizePolicy(sizePolicy)
        self.zoomSlider.setFixedHeight(60)
        self.zoomSlider.setFixedWidth(400)
        self.zoomSlider.setContentsMargins(0, 0, 0, 0)
        

        self.toggle_buttons_widget = QWidget(MainWindow)
        self.toggleBtn = ToggleButtonPanel(MainWindow)
        self.toggle_buttons_widget.setLayout(self.toggleBtn.PanelGridLayout)
        self.toggle_buttons_widget.setContentsMargins(0, 0, 0, 0) # left, top, right, bottom margins.
        self.toggle_buttons_widget.setFixedHeight(70)
        self.toggle_buttons_widget.setFixedWidth(300)


        self.CanvasUtility_layout.addWidget(self.OverviewLabel, 0, 0, 1, 1) # row, col, rowspan, colspan
        self.CanvasUtility_layout.addWidget(self.placeholder_1, 0, 1, 1, 3) # row, col, rowspan, colspan
        self.CanvasUtility_layout.addWidget(self.placeholder_1, 1, 0, 3, 4) # row, col, rowspan, colspan
        self.CanvasUtility_layout.addWidget(self.toggle_buttons_widget, 4, 1, 1, 1) # row, col, rowspan, colspan
        self.CanvasUtility_layout.addWidget(self.zoomSlider, 4, 2, 1, 2) # row, col, rowspan, colspan
        self.CanvasUtility_layout.addWidget(self.rotation_dial, 4, 4, 1, 1) # row, col, rowspan, colspan

        
        self.MainCanvas_layout.addWidget(self.CanvasUtility_parentWidget)

        self.toolBar = ToolBar(MainWindow)
        self.mainImageGridLayout.addWidget(self.toolBar, 0, 0, 1, 2)

        self.mainImageGridLayout.addLayout(self.MainCanvas_layout, 1, 0, 1, 1) # row, col, rowspan, colspan



        self.verticalScrollBar = QScrollBar(self.centralwidget)
        self.verticalScrollBar.setOrientation(Qt.Vertical)
        self.verticalScrollBar.setObjectName("verticalScrollBar")
        self.mainImageGridLayout.addWidget(self.verticalScrollBar, 1, 1, 1, 1) # row, col, rowspan, colspan

        self.horizontalScrollBar = QScrollBar(self.centralwidget)
        self.horizontalScrollBar.setMaximum(999)
        self.horizontalScrollBar.setOrientation(Qt.Horizontal)
        self.horizontalScrollBar.setObjectName("horizontalScrollBar")
        self.horizontalScrollBar.setContentsMargins(0, 0, 0, 0)
        self.mainImageGridLayout.addWidget(self.horizontalScrollBar, 2, 0, 1, 1) # row, col, rowspan, colspan
        

        self.mainImageGrid_parentWidget = QWidget()
        self.mainImageGrid_parentWidget.setLayout(self.mainImageGridLayout)
        self.horizontalSplitter.addWidget(self.mainImageGrid_parentWidget)

        self.statusBar = StatusBar(MainWindow)
        self.mainImageGridLayout.addWidget(self.statusBar, 3, 0, 1, 2) # row, col, rowspan, colspan

        self.web = QWebEngineView(self.horizontalSplitter)
        from PySide6.QtWebEngineCore import QWebEngineSettings
        self.web.settings().setAttribute(QWebEngineSettings.LocalContentCanAccessRemoteUrls, True)
        
        self.web.setObjectName(u"web")
        self.horizontalSplitter.addWidget(self.web)
        self.web.setMinimumWidth(300)
        self.web.setMaximumWidth(1000)
        self.web.setContentsMargins(0, 0, 0, 0)
        
        self.horizontalSplitter.setSizes([800,400])

        self.gridLayout.addWidget(self.horizontalSplitter)

        MainWindow.setCentralWidget(self.centralwidget)


        
        self.actionOpen = QAction(MainWindow)
        self.actionOpen.setObjectName(u"actionOpen")
        self.actionOpenCase = QAction(MainWindow)
        self.actionOpenCase.setObjectName(u"actionOpenCase")
        
        self.actionExit = QAction(MainWindow)
        self.actionExit.setObjectName(u"actionExit")

        self.menuBar = QMenuBar(MainWindow)
        self.menuBar.setObjectName(u"menubar")
        self.menuBar_width = 800
        self.menuBar_height = 26
        self.menuBar.setGeometry(QRect(0, 0, self.menuBar_width, self.menuBar_height)) # x, y, w, h
        self.menu_File = QMenu(self.menuBar)
        self.menu_File.setObjectName(u"menu_File")
        MainWindow.setMenuBar(self.menuBar)

        
        self.menuBar.addAction(self.menu_File.menuAction())
        self.menu_File.addAction(self.actionOpen)
        self.menu_File.addAction(self.actionOpenCase)
        self.menu_File.addSeparator()
        self.menu_File.addAction(self.actionExit)




        self.retranslateUI(MainWindow)

        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUI(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"nuclei.io", None))
        self.actionOpen.setText(QCoreApplication.translate("MainWindow", u"&Open slide", None))
        self.actionOpen.setToolTip(QCoreApplication.translate("MainWindow", u"Open slide", None))
        self.actionOpen.setShortcut(QCoreApplication.translate("MainWindow", u"Ctrl+O", None))

        self.actionOpenCase.setText(QCoreApplication.translate("MainWindow", u"Open case &Folder", None))
        self.actionOpenCase.setToolTip(QCoreApplication.translate("MainWindow", u"Open case folder", None))
        self.actionOpenCase.setShortcut(QCoreApplication.translate("MainWindow", u"Ctrl+F", None))

        self.actionExit.setText(QCoreApplication.translate("MainWindow", u"E&xit", None))
        self.actionExit.setToolTip(QCoreApplication.translate("MainWindow", u"Exit editor", None))
        self.actionExit.setShortcut(QCoreApplication.translate("MainWindow", u"Ctrl+Q", None))
        self.menu_File.setTitle(QCoreApplication.translate("MainWindow", u"&File", None))


        
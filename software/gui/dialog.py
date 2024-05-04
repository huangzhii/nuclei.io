#############################################################################
##
## Copyright (C) 2022 nuclei.io
## Contact: https://www.nuclei.io/
##
## This script author: Zhi Huang
## This script contributors: Zhi Huang
## This script created on: 08/13/2022
#############################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect, Slot, Signal,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QAction, QBrush, QColor, QConicalGradient,
    QCursor, QFont, QFontDatabase, QGradient,
    QIcon, QImage, QKeySequence, QLinearGradient,
    QPainter, QPalette, QPixmap, QRadialGradient,
    QTransform)
from PySide6.QtWebChannel import QWebChannel
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWidgets import (QApplication, QHBoxLayout, QVBoxLayout, QGridLayout, QStackedLayout, QMainWindow, QMenu, QLabel,
    QMenuBar, QPlainTextEdit, QSizePolicy, QSplitter, QScrollBar, QProgressBar, QPushButton, QDial, QToolBar,
    QStatusBar, QWidget, QDialog)


import sys
import os
import json
import webbrowser


class Backend_Dialog(QObject):
    #tutorial:https://www.cnblogs.com/aloe-n/p/10052830.html
    sendjson_to_js = Signal(str)

    def __init__(self, Dialog=None, engine=None, MainWindow=None):
        super().__init__(engine)
        self.Dialog = Dialog
        self.MainWindow = MainWindow

    @Slot(str)
    def log(self, logstring):
        print(logstring)

    @Slot()
    def open_local_slide_dialog(self):
        self.MainWindow.onFileOpen()

    @Slot(str)
    def openlink(self, link):
        # open link from webengine on your system's default browser.
        webbrowser.open_new(link)

    @Slot(str)
    def update_user_profile(self, jsonstring):
        jsondata = json.loads(jsonstring)
        self.MainWindow.user.login(jsondata)
        # close dialog
        self.Dialog.close()


class WebDialog(QDialog):
    def __init__(self, MainWindow):
        super().__init__()
        self.MainWindow = MainWindow
        self.setWindowTitle("nuclei.io")

    def show_dialog(self, mode: str):
        if mode == 'login':
            self.resize(400, 600)
            self.layout = QVBoxLayout()
            self.layout.setContentsMargins(0, 0, 0, 0)

            self.web_dialog = QWebEngineView()
            from PySide6.QtWebEngineCore import QWebEngineSettings
            self.web_dialog.settings().setAttribute(QWebEngineSettings.LocalContentCanAccessRemoteUrls, True)
            self.web_dialog.setContextMenuPolicy(Qt.NoContextMenu)

            self.backend_dialog = Backend_Dialog(self, self.web_dialog, self.MainWindow)
            self._channel = QWebChannel(self)
            self.web_dialog.page().setWebChannel(self._channel)
            self._channel.registerObject("backend", self.backend_dialog)

            filename = os.path.join(self.MainWindow.wd, 'HTML5_UI', 'html', 'login.html')
            url = QUrl.fromLocalFile(filename)
            self.web_dialog.load(url)
            self.layout.addWidget(self.web_dialog)
            self.setLayout(self.layout)

            self.show()


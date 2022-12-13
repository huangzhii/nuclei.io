#############################################################################
##
## Copyright (C) 2022 nuclei.io
## Contact: https://www.nuclei.io/
##
## This script author: Zhi Huang
## This script contributors: Zhi Huang
## This script created on: 07/28/2022
#############################################################################

from PySide6.QtCore import QDir, QFile, QIODevice, QUrl, Qt, Slot
from PySide6.QtGui import QDesktopServices
from PySide6.QtWebEngineCore import QWebEnginePage
from PySide6.QtWebChannel import QWebChannel
import os, sys

from webengine.backend import Backend



def webengine_setup(self):
    self.ui.web.setContextMenuPolicy(Qt.NoContextMenu)

    backend = Backend(self.ui.web, self)
    self._channel = QWebChannel(self)
    self.ui.web.page().setWebChannel(self._channel)
    self._channel.registerObject("backend", backend)

    #self.ui.web.setUrl(QUrl("qrc:/index.html"))
    filename = os.path.join(self.wd, 'HTML5_UI', 'index.html')
    url = QUrl.fromLocalFile(filename)
    
    if self.CHROME_DEBUG:
        QDesktopServices.openUrl(url)
    else:
        self.ui.web.load(url)

    return backend
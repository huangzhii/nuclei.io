#############################################################################
##
## Copyright (C) 2022 nuclei.io
## Contact: https://www.nuclei.io/
##
## This script author: Zhi Huang
## This script contributors: Zhi Huang
## This script created on: 07/25/2022
#############################################################################


import sys
import os
opj = os.path.join
from PySide6.QtCore import QCoreApplication
from PySide6.QtWidgets import (QApplication, QSystemTrayIcon)

from PySide6.QtGui import (QAction, QBrush, QColor, QConicalGradient,
    QCursor, QFont, QFontDatabase, QGradient,
    QIcon, QImage, QKeySequence, QLinearGradient,
    QPainter, QPalette, QPixmap, QRadialGradient,
    QNativeGestureEvent,
    QTransform)

from mainwindow import MainWindow
import multiprocess
import multiprocessing
'''
Many browsers for security reason disable the loading of local files.
We need --disable-web-security, otherwise, Bootstrap CSS and JS won't load.
'''

sys.argv.append("--disable-web-security")
sys.argv.append("--allow-file-access-from-files")


if __name__ == '__main__':
    multiprocess.freeze_support() # this is for packaging nuclei.app software in Mac
    multiprocessing.freeze_support()
    
    app = QApplication(sys.argv)
    # if use debug mode, make sure wd is under software/ folder.
    wd = os.path.realpath('.')

    if 'software' not in wd:
        wd = os.path.join(wd, 'software')
    
    icon = QIcon(opj(wd, 'software', 'Artwork', 'icon', 'icon.icns'))
    app.setWindowIcon(icon)

    # Create the tray
    tray = QSystemTrayIcon()
    tray.setIcon(icon)
    tray.setVisible(True)


    QCoreApplication.setOrganizationName("nuclei.io")
    window = MainWindow(wd)
    window.show()
    sys.exit(app.exec())

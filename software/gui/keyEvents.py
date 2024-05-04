#############################################################################
##
## Copyright (C) 2022 nuclei.io
## Contact: https://www.nuclei.io/
##
## This script author: Zhi Huang
## This script contributors: Zhi Huang
## This script created on: 07/31/2022
#############################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)

from PySide6.QtGui import (QAction, QBrush, QColor, QConicalGradient,
    QCursor, QFont, QFontDatabase, QGradient, QWheelEvent,
    QIcon, QImage, QKeySequence, QLinearGradient,
    QPainter, QPalette, QPixmap, QRadialGradient,
    QTransform)

from PySide6.QtWidgets import (QApplication, QHBoxLayout, QGridLayout, QStackedLayout, QMainWindow, QMenu, QLabel,
    QMenuBar, QPlainTextEdit, QSizePolicy, QSplitter, QScrollBar, QProgressBar, QPushButton, QDial,
    QStatusBar, QWidget)
from gui import show_image
import numpy as np

def whenKeyPressed(self, event):
    print('Keyboard input *** key pressed: key: %s; text: "%s"' % (event.key(), event.text()))
    self.log.write('Keyboard input *** key pressed: key: %s; text: "%s"' % (event.key(), event.text()))
    if event.text() == 'z': # press m key to show ML overlay
        if self.ui.toggleBtn.toggle_ml.checkState() == Qt.CheckState.Unchecked:
            self.ui.toggleBtn.toggle_ml.setCheckState(Qt.CheckState.Checked)
        else:
            self.ui.toggleBtn.toggle_ml.setCheckState(Qt.CheckState.Unchecked)
        self.toggleOverlay('ML Overlay')

    if event.text() == 'x': # press n key to show annotation
        if self.ui.toggleBtn.toggle_a.checkState() == Qt.CheckState.Unchecked:
            self.ui.toggleBtn.toggle_a.setCheckState(Qt.CheckState.Checked)
        else:
            self.ui.toggleBtn.toggle_a.setCheckState(Qt.CheckState.Unchecked)
        self.toggleOverlay('Annotation Overlay')
        #print('self.toggle_Annotation_Overlay',self.toggle_Annotation_Overlay)
        #if self.toggle_Annotation_Overlay: ischecked = False
        #else: ischecked = True
        #print('ischecked',ischecked)
        #self.toggleOverlay('Annotation Overlay', ischecked)


    if event.key() == Qt.Key_Control: # press space key to show overlay
        print('Ctrl pressed')
        self.is_key_ctrl_pressed = True
        QApplication.setOverrideCursor(Qt.CrossCursor)
        #self.ui.mode = UIMainMode.MODE_ANNOTATE_AREA
        #self.menuItemAnnotateArea.trigger()
    

    if event.key() == Qt.Key_Escape: # press space key to show overlay
        print('Key Escape pressed')
        if (self.ui.mode == 'Polygon'):
            self.draw.cancelPolygon()


    if event.text() in ['w', 'a', 's', 'd']:
        imgarea_p1 = self.region[0]
        imgarea_w =  self.region[1]
        imgCenter = (np.int64(imgarea_p1[0] + imgarea_w[0]/2), np.int64(imgarea_p1[1] + imgarea_w[1]/2))
        movestep = 300*self.getZoomValue()
        if event.text() == 'w': # press w key to move up
            newCenter = (imgCenter[0], imgCenter[1]-movestep)
        if event.text() == 'a':
            newCenter = (imgCenter[0]-movestep, imgCenter[1])
        if event.text() == 's':
            newCenter = (imgCenter[0], imgCenter[1]+movestep)
        if event.text() == 'd':
            newCenter = (imgCenter[0]+movestep, imgCenter[1])
        self.setCenter(newCenter)
        #self.update


def whenKeyReleased(self, event):
    if event.key() == Qt.Key_Control:
        self.is_key_ctrl_pressed = False
        '''
        self.ui.mode = UIMainMode.MODE_VIEW
        self.menuItemView.trigger()
        '''
        QApplication.setOverrideCursor(Qt.ArrowCursor)
        print('Ctrl released.')

        

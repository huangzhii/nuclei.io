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
    QMetaObject, QObject, QPoint, QRect, Signal, QLineF, QEvent,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QAction, QBrush, QColor, QConicalGradient,
    QCursor, QFont, QFontDatabase, QGradient, QMouseEvent,
    QIcon, QImage, QKeySequence, QLinearGradient,
    QPainter, QPalette, QPixmap, QRadialGradient,
    QTransform)
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWidgets import (QApplication, QHBoxLayout, QVBoxLayout, QGridLayout, QMainWindow, QMenu, QLabel,
    QMenuBar, QPlainTextEdit, QSizePolicy, QSplitter, QScrollBar, QProgressBar, QPushButton, QDial,
    QStatusBar, QWidget)

class ValueDial(QWidget):
    _dialProperties = ('minimum', 'maximum', 'value', 'singleStep', 'pageStep',
        'notchesVisible', 'tracking', 'wrapping', 
        'invertedAppearance', 'invertedControls', 'orientation')
    _inPadding = 3
    _outPadding = 2
    valueChanged = Signal(int)

    def __init__(self, *args, **kwargs):
        # remove properties used as keyword arguments for the dial
        dialArgs = {k:v for k, v in kwargs.items() if k in self._dialProperties}
        for k in dialArgs.keys():
            kwargs.pop(k)
        super().__init__(*args, **kwargs)
        layout = QVBoxLayout(self)
        self.dial = QDial(self, **dialArgs)
        layout.addWidget(self.dial)
        self.dial.valueChanged.connect(self.valueChanged)
        # make the dial the focus proxy (so that it captures focus *and* key events)
        self.setFocusProxy(self.dial)

        # simple "monkey patching" to access dial functions
        self.value = self.dial.value
        self.setValue = self.dial.setValue
        self.minimum = self.dial.minimum
        self.maximum = self.dial.maximum
        self.wrapping = self.dial.wrapping
        self.notchesVisible = self.dial.notchesVisible
        self.setNotchesVisible = self.dial.setNotchesVisible
        self.setNotchTarget = self.dial.setNotchTarget
        self.notchSize = self.dial.notchSize
        self.invertedAppearance = self.dial.invertedAppearance
        self.setInvertedAppearance = self.dial.setInvertedAppearance

        self.updateSize()

    def inPadding(self):
        return self._inPadding

    def setInPadding(self, padding):
        self._inPadding = max(0, padding)
        self.updateSize()

    def outPadding(self):
        return self._outPadding

    def setOutPadding(self, padding):
        self._outPadding = max(0, padding)
        self.updateSize()

    # the following functions are required to correctly update the layout
    def setMinimum(self, minimum):
        self.dial.setMinimum(minimum)
        self.updateSize()

    def setMaximum(self, maximum):
        self.dial.setMaximum(maximum)
        self.updateSize()

    def setWrapping(self, wrapping):
        self.dial.setWrapping(wrapping)
        self.updateSize()

    def updateSize(self):
        # a function that sets the margins to ensure that the value strings always
        # have enough space
        fm = self.fontMetrics()
        minWidth = max(fm.boundingRect(str(v*15)).width() for v in range(self.minimum(), self.maximum() + 1))
        self.offset = max(minWidth, fm.height()) / 2
        margin = self.offset + self._inPadding + self._outPadding
        self.layout().setContentsMargins(margin, margin, margin, margin)

    def translateMouseEvent(self, event):
        # a helper function to translate mouse events to the dial
        return QMouseEvent(event.type(), 
            self.dial.mapFrom(self, event.pos()), 
            event.button(), event.buttons(), event.modifiers())

    def changeEvent(self, event):
        if event.type() == QEvent.FontChange:
            self.updateSize()

    def mousePressEvent(self, event):
        self.dial.mousePressEvent(self.translateMouseEvent(event))

    def mouseMoveEvent(self, event):
        self.dial.mouseMoveEvent(self.translateMouseEvent(event))

    def mouseReleaseEvent(self, event):
        self.dial.mouseReleaseEvent(self.translateMouseEvent(event))

    def paintEvent(self, event):
        radius = min(self.width(), self.height()) / 2
        radius -= (self.offset / 2 + self._outPadding)
        invert = -1 if self.invertedAppearance() else 1
        if self.wrapping():
            angleRange = 360
            startAngle = 270
            rangeOffset = 0
        else:
            angleRange = 300
            startAngle = 240 if invert > 0 else 300
            rangeOffset = 1
        fm = self.fontMetrics()

        # a reference line used for the target of the text rectangle
        reference = QLineF.fromPolar(radius, 0).translated(self.rect().center())
        fullRange = self.maximum() - self.minimum()
        textRect = QRect()

        qp = QPainter(self)
        qp.setRenderHints(QPainter.Antialiasing)
        for p in range(0, fullRange + rangeOffset, self.notchSize()):
            value = self.minimum() + p
            value_to_360 = value*15
            if invert < 0:
                value -= 1
                if value < self.minimum():
                    continue
            angle = p / fullRange * angleRange * invert
            reference.setAngle(startAngle - angle)
            textRect.setSize(fm.size(Qt.TextSingleLine, str(value_to_360)))
            textRect.moveCenter(reference.p2().toPoint())


            if value_to_360 % 45 == 0:
                qp.drawText(textRect, Qt.AlignCenter, str(value_to_360))
            else:
                pass
    
    def value_changed(self, MainWindow, value):
        MainWindow.rotation = value * 15
        print('Dial angle:', MainWindow.rotation)

        '''
        self.rotation = True
        if angle > 180:
            self.rotate=True
        else:
            self.rotate=False
        '''

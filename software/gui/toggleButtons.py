
from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect, Signal, QLineF, QEvent,
    QSize, QTime, QUrl, Qt, QEasingCurve, QPropertyAnimation, Property)
from PySide6.QtGui import (QAction, QBrush, QColor, QConicalGradient,
    QCursor, QFont, QFontDatabase, QGradient, QMouseEvent,
    QIcon, QImage, QKeySequence, QLinearGradient,
    QPainter, QPalette, QPixmap, QRadialGradient,
    QTransform)
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWidgets import (QApplication, QHBoxLayout, QVBoxLayout, QGridLayout, QMainWindow, QMenu, QLabel,
    QMenuBar, QPlainTextEdit, QSizePolicy, QSplitter, QScrollBar, QProgressBar, QPushButton, QDial,
    QStatusBar, QWidget, QCheckBox)


class PyToggle(QCheckBox):
    def __init__(self,
                MainWindow,
                toggle_name = '',
                width = 60,
                height = 28,
                bg_color = "#777",
                circle_color = "#DDD",
                active_color = "#00b551",
                animation_curve = QEasingCurve.OutBounce):
        QCheckBox.__init__(self)

        self.MainWindow = MainWindow
        self.toggle_name = toggle_name
        # Set default parameters
        self.setFixedSize(width, height)
        self.setCursor(Qt.PointingHandCursor)

        # Colors
        self._bg_color = bg_color
        self._circle_color = circle_color
        self._active_color = active_color

        # Create animation
        self._circle_position = 3
        
        self.animation = QPropertyAnimation(self, b"circle_position", self)
        self.animation.setEasingCurve(animation_curve)
        self.animation.setDuration(500)

        # Connect state changed
        self.stateChanged.connect(self.start_transition)
    
    # Create new set and get properties
    @Property(float)
    def circle_position(self):
        return self._circle_position
    
    @circle_position.setter
    def circle_position(self, pos):
        self._circle_position = pos
        self.update()

    def start_transition(self, value):
        self.animation.stop()
        if value:
            self.animation.setEndValue(self.width() - 26)
        else:
            self.animation.setEndValue(3)
        
        # Start animation
        self.animation.start()
        self.MainWindow.toggleOverlay(self.toggle_name)
        #print('status', self.isChecked())

    # Set new hit area
    def hitButton(self, pos: QPoint) -> bool:
        return self.contentsRect().contains(pos)

    # Draw new items
    def paintEvent(self, e):
        # Set painter
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)

        # Set as no pen
        p.setPen(Qt.NoPen)

        # Draw rectangle
        rect = QRect(0,0, self.width(), self.height())

        if not self.isChecked():
            # Draw BG
            p.setBrush(QColor(self._bg_color))
            p.drawRoundedRect(0,0, rect.width(), self.height(), self.height()/2, self.height()/2)

            # Draw Circle
            p.setBrush(QColor(self._circle_color))
            p.drawEllipse(3,3,22,22)
        else:
            # Draw BG
            p.setBrush(QColor(self._active_color))
            p.drawRoundedRect(0,0, rect.width(), self.height(), self.height()/2, self.height()/2)

            # Draw Circle
            p.setBrush(QColor(self._circle_color))
            p.drawEllipse(self.width()-26,3,22,22)
            
        # End draw
        p.end()


class ToggleButtonPanel(QWidget):


    def __init__(self,
                MainWindowObject):      
        super(ToggleButtonPanel, self).__init__()
        self.MainWindow = MainWindowObject
        self.initUI()



    def initUI(self):

        self.PanelGridLayout = QGridLayout()
        self.PanelGridLayout.setObjectName("verticalLayout")
        self.PanelGridLayout.setContentsMargins(0, 0, 0, 0) # left, top, right, bottom margins.
        self.PanelGridLayout.setSpacing(0)

        self.toggle_ml = PyToggle(self.MainWindow, 'ML overlay')
        self.toggle_ml.setContentsMargins(0, 2, 0, 2) # left, top, right, bottom margins.
        self.text_ML_overlay = QLabel()
        self.text_ML_overlay.setText('ML overlay (Z)')
        self.text_ML_overlay.setContentsMargins(10, 0, 0, 0) # left, top, right, bottom margins.


        self.toggle_a = PyToggle(self.MainWindow, 'Annotation overlay')
        self.toggle_a.setContentsMargins(0, 2, 0, 2) # left, top, right, bottom margins.
        self.text_Annotation_overlay = QLabel()
        self.text_Annotation_overlay.setText('Annotation overlay (X)')
        self.text_Annotation_overlay.setContentsMargins(10, 0, 0, 0) # left, top, right, bottom margins.


        self.PanelGridLayout.addWidget(self.toggle_ml, 0, 0, 1, 1)
        self.PanelGridLayout.addWidget(self.text_ML_overlay, 0, 1, 1, 1)

        self.PanelGridLayout.addWidget(self.toggle_a, 1, 0, 1, 1)
        self.PanelGridLayout.addWidget(self.text_Annotation_overlay, 1, 1, 1, 1)


        return self.PanelGridLayout
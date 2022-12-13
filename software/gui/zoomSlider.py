from PySide6.QtCore import Signal, Qt
from PySide6.QtWidgets import QWidget
from PySide6.QtGui import QPainter, QColor, QFont
import numpy as np

from gui import show_image
from functools import partial

class ZoomSlider(QWidget):

    maxZoom = 0.3
    minZoom = 80
    sliderPos = 0
    zoomLevel = 0
    valueChanged = Signal()
    text = 'N/A'

  
    def __init__(self, MainWindowObject):      
        super(ZoomSlider, self).__init__()
        self.MainWindow = MainWindowObject
        
        self.initUI()

    def initUI(self):
        
        self.setMinimumSize(1, 40)
        self.value = 700
        self.num = list()
        self.setSteps()
        self.valueChanged.connect(self.sliderChanged)


    def sliderToZoomValue(self, value):
        return np.power(2,value/100*(np.log2(0.25/ self.getMaxZoom())))*self.getMaxZoom()

    def zoomValueToSlider(self, value : float) -> float:
        maxZoom = self.getMaxZoom()
        retval = 100*np.log2(value/(maxZoom))/(np.log2(0.25/maxZoom))
        return 100*np.log2(value/(maxZoom))/(np.log2(0.25/maxZoom))

    def setMaxZoom(self, maxZoom : float):
        self.maxZoom = maxZoom
        self.setSteps()
        self.repaint()

    def getMaxZoom(self) -> float:
        return self.maxZoom
    
    def setText(self, text:str):
        self.text = text
        self.repaint()

    def getValue(self) -> float:
        return self.sliderPos
    
    def setValue(self, value:float):
        self.sliderPos = value
        self.zoomLevel = self.sliderToZoomValue(self.sliderPos)
        self.repaint()


    def setMinZoom(self, value : float):
        self.minZoom = value
        self.setSteps()
        self.repaint()

    def mouseMoveEvent(self, event):
        self.mousePressEvent(event, dragging=True)
    
    def mouseReleaseEvent(self, event):
        self.mousePressEvent(event, dragging=True)
        self.valueChanged.emit()

    def mousePressEvent(self, event, dragging=False):
        if ((event.button() == Qt.LeftButton)) or dragging:
            pntr_x = float(event.localPos().x())
            w = self.size().width() - 40

            zoomPos = (pntr_x-20)/w
            self.sliderPos = 100*zoomPos

            self.sliderPos = np.clip(self.sliderPos,0,100)
            self.zoomLevel = self.sliderToZoomValue(self.sliderPos)

            self.repaint()
            


    def setSteps(self):
        zoomList = 1 / np.float_power(2,np.asarray([-7,-6,-5,-4,-3,-2,-1,0]))
        self.steps = []
        self.num = []
        for step in zoomList:
            self.steps.append(self.minZoom / step / 2)
            self.num.append(self.zoomValueToSlider(step))
        self.steps.append(self.minZoom)
        self.num.append(self.zoomValueToSlider(0.5))



    def paintEvent(self, e):
        qp = QPainter()
        qp.begin(self)
        self.drawWidget(qp)
        qp.end()
    
    
      
    def drawWidget(self,
                    qp:QPainter):
        metrics = qp.fontMetrics()

        size = self.size()
        w = size.width()
        h = size.height()

        step = int(round(w / 10.0))

        till = int(((w / 750.0) * self.value))
        full = int(((w / 750.0) * 700))

        w_net = w - 40
        y_offset = 30
        qp.setBrush(QColor(93, 93, 93))
        qp.setPen(QColor(39, 39, 39))
        qp.drawRect(20-1, y_offset, w_net+2, 5)

        qp.setBrush(QColor(70, 70, 70))
        font = QFont('Arial', 8, QFont.Black)
        qp.setFont(font)

        for j in range(len(self.steps)):
            qp.setPen(QColor(93, 93, 93))
            qp.drawLine(self.num[j]*w_net/100+20, y_offset+10, self.num[j]*w_net/100+20, 32)
            labelstr = str(self.steps[j])+' x'
            fw = metrics.boundingRect(self.text).width()
            qp.setPen(QColor(0, 0, 0))
            qp.drawText(self.num[j]*w_net/100-fw/2+20, y_offset+25, labelstr)


        font = QFont('Arial', 12, QFont.Black)
        qp.setFont(font)


        tw = metrics.boundingRect(self.text).width()
        qp.setPen(QColor(0, 125, 255))
        qp.setBrush(QColor(0, 125, 255))

        qp.drawRect(w_net*self.sliderPos*0.01+20-5, y_offset-7, 10, 20)
        qp.setPen(QColor(0, 125, 255))
        qp.drawRect(w_net*self.sliderPos*0.01+20-4, y_offset-6, 8, 20-2)

        qp.setBrush(QColor(70, 70, 70))
        qp.setPen(QColor(0, 0, 0))
        qp.drawText(self.sliderPos*w_net/100-tw/2+20, y_offset-15, self.text)


    def sliderChanged(self):
        """
            Callback function for when a slider was changed.
        """
        
        if self.MainWindow.imageOpened:
            print('Slider changed')
            self.MainWindow.setZoomValue(self.sliderToZoomValue(self.sliderPos))
            self.MainWindow.showImage()
            self.MainWindow.updateScrollbars()
        else:
            print('No image open. Slider change won\'t work.')
            pass

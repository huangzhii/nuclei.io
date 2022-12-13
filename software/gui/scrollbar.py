
#############################################################################
##
## Copyright (C) 2022 nuclei.io
## Contact: https://www.nuclei.io/
##
## This script author: Zhi Huang
## This script contributors: Zhi Huang
## This script created on: 07/30/2022
#############################################################################

from gui import show_image

def changeScrollbars(self, value=None):
    """
        Callback function when the scrollbars (horizontal/vertical) are changed.
    """
    if not (self.imageOpened):
        return

    if (self.imageOpened):
        self.relativeCoords[0] = (self.ui.horizontalScrollBar.value()/self.ui.hsteps)-0.5
        self.relativeCoords[1] = (self.ui.verticalScrollBar.value()/self.ui.vsteps)-0.5
        self.showImage()

def updateScrollbars(self, value=None):
    """
        Update the scrollbars when the position was changed by another method.
    """
    if not (self.imageOpened):
        return

    try:
        self.ui.horizontalScrollBar.valueChanged.disconnect()
        self.ui.verticalScrollBar.valueChanged.disconnect()
    except Exception as e:
        print(e)
    viewsize = self.mainImageSize * self.getZoomValue()

    self.ui.horizontalScrollBar.setMaximum(0)
    self.ui.hsteps=int(10*self.slide.level_dimensions[0][0]/viewsize[0])
    self.ui.vsteps=int(10*self.slide.level_dimensions[0][1]/viewsize[1])
    self.ui.horizontalScrollBar.setMaximum(self.ui.hsteps)
    self.ui.horizontalScrollBar.setMinimum(0)

    self.ui.verticalScrollBar.setMaximum(self.ui.vsteps)
    self.ui.verticalScrollBar.setMinimum(0)

    self.ui.horizontalScrollBar.setValue(int((self.relativeCoords[0]+0.5)*self.ui.hsteps))
    self.ui.verticalScrollBar.setValue(int((self.relativeCoords[1]+0.5)*self.ui.vsteps))

    self.ui.horizontalScrollBar.valueChanged.connect(self.changeScrollbars)
    self.ui.verticalScrollBar.valueChanged.connect(self.changeScrollbars)

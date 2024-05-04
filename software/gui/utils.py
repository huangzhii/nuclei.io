

#############################################################################
##
## Copyright (C) 2022 nuclei.io
## Contact: https://www.nuclei.io/
##
## This script author: Zhi Huang
## This script contributors: Zhi Huang
## This script created on: 08/09/2022
#############################################################################

from re import S
from PySide6.QtCore import (QDir, QFile, QIODevice, QUrl, Qt, Slot, Signal,
                            QPoint, QPointF)
from PySide6.QtGui import (QColor, QBrush, QPainter, QPainterPath, QPen,
                           QPolygonF, QPolygon, QRadialGradient)

import shiboken6, ctypes
import numpy as np


QT_API = 'pyside6'
def array2d_to_qpolygonf(xdata, ydata):
    """
    Utility function to convert two 1D-NumPy arrays representing curve data
    (X-axis, Y-axis data) into a single polyline (QtGui.PolygonF object).
    This feature is compatible with PyQt4, PyQt5 and PySide2 (requires QtPy).
    License/copyright: MIT License © Pierre Raybaut 2020-2021.
    :param numpy.ndarray xdata: 1D-NumPy array
    :param numpy.ndarray ydata: 1D-NumPy array
    :return: Polyline
    :rtype: QtGui.QPolygonF
    """
    if not (xdata.size == ydata.size == xdata.shape[0] == ydata.shape[0]):
        raise ValueError("Arguments must be 1D NumPy arrays with same size")
    size = xdata.size
    if QT_API.startswith("pyside"):  # PySide (obviously...)
        if QT_API == "pyside2":
            polyline = QPolygonF(size)
        else:
            polyline = QPolygonF()
            polyline.resize(size)
        address = shiboken6.getCppPointer(polyline.data())[0]
        buffer = (ctypes.c_double * 2 * size).from_address(address)
    else:  # PyQt4, PyQt5
        if QT_API == "pyqt6":
            polyline = QPolygonF([QPointF(0, 0)] * size)
        else:
            polyline = QPolygonF(size)
        buffer = polyline.data()
        buffer.setsize(16 * size)  # 16 bytes per point: 8 bytes per X,Y value (float64)
    memory = np.frombuffer(buffer, np.float64)
    memory[: (size - 1) * 2 + 1 : 2] = np.array(xdata, dtype=np.float64, copy=False)
    memory[1 : (size - 1) * 2 + 2 : 2] = np.array(ydata, dtype=np.float64, copy=False)
    return polyline





import base64
from io import BytesIO
# Convert Image to Base64 
def im_2_b64(image):
    buff = BytesIO()
    image.save(buff, format="JPEG")
    img_str = base64.b64encode(buff.getvalue())
    return img_str
    
def np_2_b64(np_array):
    '''
    Encode:
    s = base64.b64encode(np_array)

    Decode:
    r = base64.decodebytes(s)
    q = np.frombuffer(r, dtype=np.uint8)

    '''
    s = base64.b64encode(np_array)
    return s


'''
import multiprocess as mp
mp.freeze_support() # this is for packaging nuclei.app software.
def parfun(f, q_in, q_out):
    while True:
        i, x = q_in.get()
        if i is None:
            break
        q_out.put((i, f(x)))

def parmap(f, X, nprocs=mp.cpu_count()):
    q_in = mp.Queue(1)
    q_out = mp.Queue()
    proc = [mp.Process(target=parfun, args=(f, q_in, q_out)) for _ in range(nprocs)]
    for p in proc:
        p.daemon = True
        p.start()
    sent = [q_in.put((i, x)) for i, x in enumerate(X)]
    [q_in.put((None, None)) for _ in range(nprocs)]
    res = [q_out.get() for _ in range(len(sent))]
    [p.join() for p in proc]
    return [x for i, x in sorted(res)]
'''
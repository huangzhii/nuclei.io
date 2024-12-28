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
    QStatusBar, QWidget, QDialog, QComboBox, QDialogButtonBox)


import sys
import os
import json
import webbrowser
import numpy as np
from PIL import Image

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

class IHC_Evaulation_Dialog:
    def __init__(self, main_window):
        self.MainWindow = main_window
        self.tissue_list = ['adipose', 'adrenal gland', 'appendix', 'bone marrow', 'breast',
                        'bronchus', 'carcinoid', 'caudate', 'cerebellum',
                        'cerebral cortex', 'cervical', 'cervix', 'colon', 'colorectal',
                        'duodenum', 'endometrial', 'endometrium', 'epididymis',
                        'esophagus', 'fallopian tube', 'gallbladder', 'glioma',
                        'head and neck', 'heart muscle', 'hippocampus', 'kidney', 'liver',
                        'lung', 'lymph node', 'lymphoma', 'melanoma', 'nasopharynx',
                        'oral mucosa', 'ovarian', 'ovary', 'pancreas', 'pancreatic',
                        'parathyroid gland', 'placenta', 'prostate', 'rectum', 'renal',
                        'salivary gland', 'seminal vesicle', 'skeletal muscle', 'skin',
                        'small intestine', 'smooth muscle', 'soft', 'spleen', 'stomach',
                        'testis', 'thyroid', 'thyroid gland', 'tonsil', 'urinary bladder',
                        'urothelial', 'vagina'] # N = 58 (Pancreas and Pancreatic cancer are merged. If not perged, N = 65)
        
        self.cell_type_list = ['Glandular cells', 'Exocrine glandular cells', 'Tumor cells',
                            'Cholangiocytes', 'Adipocytes', 'Squamous epithelial cells',
                            'Glial cells', 'Cells in endometrial stroma', 'Cells in red pulp',
                            'Alveolar cells', 'Respiratory epithelial cells',
                            'Cells in granular layer', 'Endothelial cells', 'Fibroblasts',
                            'Decidual cells', 'Cells in glomeruli', 'Myocytes',
                            'Cells in seminiferous ducts', 'Hematopoietic cells',
                            'Germinal center cells', 'Cardiomyocytes', 'Urothelial cells',
                            'Trophoblastic cells', 'Smooth muscle cells',
                            'Ovarian stroma cells', 'Follicle cells', 'Epidermal cells',
                            'Chondrocytes', 'Hepatocytes', 'Lymphoid tissue',
                            'Non-germinal center cells', 'Cells in molecular layer',
                            'Keratinocytes', 'Peripheral nerve', 'Cells in tubules',
                            'Neuronal cells', 'Leydig cells', 'Cells in white pulp', 'Langerhans'] # Dec 4 -- added remaining cell types in hpa11m

    def show_menu(self, annotation_dict):
        dialog = QDialog(self.MainWindow)
        dialog.setWindowTitle("IHC Staining Evaluation")
        layout = QVBoxLayout()

        ROI_type = annotation_dict['type']
        ROI_points_global = annotation_dict['points_global']
        if ROI_type == 'Rect':
            [[x1, y1], [x2, y2]] = ROI_points_global
            x1, x2 = np.sort([x1, x2])
            y1, y2 = np.sort([y1, y2])
            # Read region and ensure proper RGB conversion
            selected_region = self.MainWindow.slide.read_region(location=(x1,y1), level=0, size=(x2-x1,y2-y1))
            # Convert from RGBA to RGB explicitly
            selected_region = Image.fromarray(np.array(selected_region)[...,:3])  # Keep only RGB channels
            selected_region = selected_region.convert('RGBA')
            



        # Add title and subtitle
        title_label = QLabel("Protein Atlas IHC Evaluator")
        title_font = QFont()
        title_font.setPointSize(16)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)

        subtitle_label = QLabel("AI-based evaluation trained on 10M IHC images")
        subtitle_font = QFont()
        subtitle_font.setPointSize(9)
        subtitle_label.setFont(subtitle_font)
        subtitle_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(subtitle_label)

        # Add image display and dimensions
        if ROI_type == 'Rect':
            width = x2-x1
            height = y2-y1
            
            # Convert PIL Image to QPixmap and scale if needed
            selected_region = selected_region.convert('RGB')
            img_qt = QPixmap.fromImage(QImage(
                selected_region.tobytes(),
                selected_region.width,
                selected_region.height,
                QImage.Format_RGB888
            ))
            
            # Scale to fit within 400x400 while maintaining aspect ratio
            scaled_pixmap = img_qt.scaled(
                400, 400,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            
            # Create and add image label
            image_label = QLabel()
            image_label.setPixmap(scaled_pixmap)
            image_label.setAlignment(Qt.AlignCenter)
            layout.addWidget(image_label)
            
            # Add dimensions label
            dim_label = QLabel(f"Region selected: {width} × {height} pixels")
            dim_label.setAlignment(Qt.AlignCenter)
            dim_note = QLabel("Recommended region size: 1000-5000 pixels per dimension")
            dim_note.setAlignment(Qt.AlignCenter)
            layout.addWidget(dim_label)
            layout.addWidget(dim_note)

        # Add some spacing between header and content
        layout.addSpacing(20)

        # Tissue type selection
        tissue_layout = QHBoxLayout()
        tissue_label = QLabel("Tissue type:")
        tissue_combo = QComboBox()
        tissue_combo.addItems(self.tissue_list)
        tissue_input = QPlainTextEdit()
        tissue_input.setMaximumHeight(30)
        tissue_input.setPlaceholderText("or type manually")
        # Set placeholder text color to gray
        palette = tissue_input.palette()
        palette.setColor(QPalette.PlaceholderText, QColor("#808080"))
        tissue_input.setPalette(palette)
        tissue_layout.addWidget(tissue_label)
        tissue_layout.addWidget(tissue_combo)
        tissue_layout.addWidget(tissue_input)
        layout.addLayout(tissue_layout)

        # Cell type selection
        cell_layout = QHBoxLayout()
        cell_label = QLabel("Cell type:")
        cell_combo = QComboBox()
        cell_combo.addItems(self.cell_type_list)
        cell_input = QPlainTextEdit()
        cell_input.setMaximumHeight(30)
        cell_input.setPlaceholderText("or type manually")
        # Set placeholder text color to gray
        palette = cell_input.palette()
        palette.setColor(QPalette.PlaceholderText, QColor("#808080"))
        cell_input.setPalette(palette)
        cell_layout.addWidget(cell_label)
        cell_layout.addWidget(cell_combo)
        cell_layout.addWidget(cell_input)
        layout.addLayout(cell_layout)

        # Antibody name
        antibody_label = QLabel("Antibody (gene) name (optional):")
        antibody_input = QPlainTextEdit()
        antibody_input.setMaximumHeight(50)
        antibody_input.setPlaceholderText("Example: CDK1")
        palette = antibody_input.palette()
        palette.setColor(QPalette.PlaceholderText, QColor("#808080")) # Set placeholder text color to gray
        antibody_input.setPalette(palette)
        layout.addWidget(antibody_label)
        layout.addWidget(antibody_input)

        # Additional information
        additional_label = QLabel("Additional information (optional):")
        additional_input = QPlainTextEdit()
        additional_input.setMaximumHeight(100)
        additional_input.setPlaceholderText("Example: Pancreas adenocarcinoma, I stained CDK1 for tumor cells. What is your interpretation?")
        palette = additional_input.palette()
        palette.setColor(QPalette.PlaceholderText, QColor("#808080")) # Set placeholder text color to gray
        additional_input.setPalette(palette)
        layout.addWidget(additional_label)
        layout.addWidget(additional_input)

        # Add OK and Cancel buttons
        button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        button_box.button(QDialogButtonBox.Ok).setText("Run")
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        layout.addWidget(button_box)

        dialog.setLayout(layout)

        if dialog.exec() == QDialog.Accepted:
            # Get the selected/entered values
            tissue = tissue_input.toPlainText() or tissue_combo.currentText()
            cell = cell_input.toPlainText() or cell_combo.currentText()
            antibody = antibody_input.toPlainText()
            additional_info = additional_input.toPlainText()

            # Proceed with IHC evaluation using the collected parameters
            self.MainWindow.datamodel.IHC_evaluation(
                annotation_dict,
                tissue_type=tissue,
                cell_type=cell,
                antibody=antibody,
                additional_info=additional_info
            )
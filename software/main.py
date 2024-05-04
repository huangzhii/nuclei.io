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
import platform
opj = os.path.join
from PySide6.QtCore import QCoreApplication
from PySide6.QtWidgets import (QApplication, QSystemTrayIcon)
from PySide6.QtGui import (QIcon)


system = platform.system()

if system == "Darwin":
    print("Current system is macOS.")
    # multiprocessing.freeze_support() is a must-have. Otherwise the program will run in infinite loop.
    # The purpose of this function is to allow the code that uses the multiprocessing module to be packaged into a standalone executable.
    import multiprocess
    multiprocess.freeze_support() # this is for packaging nuclei.app software.
    # Even if we will not use multiprocessing, we still need to include the following two lines:
    # Because some other packages (perhaps tiffslide & PIL) will use it.
    import multiprocessing
    multiprocessing.freeze_support() # this is for packaging nuclei.app software.
    
    #For some unknown error:
    #Traceback (most recent call last):
    #    File "multiprocessing/resource_tracker.py", line 201, in main
    #KeyError: '/mp-inwxqau3'
    #In mac os, do not use multiprocessing for slideReader.py. Use multiprocess.
    
else:
    # Windows
    import multiprocess
    multiprocess.freeze_support() # this is for packaging nuclei.app software.
    import multiprocessing
    multiprocessing.freeze_support() # this is for packaging nuclei.app software.


# For mac OS, we must put the two lines "multiprocess.freeze_support()" and "multiprocessing.freeze_support()" before import mainwindow.
# Otherwise, the program will result in infinite loop running in backend, which is not noticable.
from mainwindow import MainWindow


if __name__ == '__main__':
    
    app = QApplication(sys.argv)
    '''
    Note: this freeze_support must run before:
    sys.argv.append("--disable-web-security")
    sys.argv.append("--allow-file-access-from-files")
    Otherwise, there will be a bug.
    '''

    '''
    Many browsers for security reason disable the loading of local files.
    We need --disable-web-security, otherwise, Bootstrap CSS and JS won't load.
    '''

    sys.argv.append("--disable-web-security")
    sys.argv.append("--allow-file-access-from-files")


    # if use debug mode, make sure wd is under software/ folder.

    if system == "Darwin":
        # if system is MacOS, the wd is /. which is not the way we want.
        # Get the path to the current script
        script_path = os.path.abspath(sys.argv[0])
        # Get the directory containing the script
        script_dir = os.path.dirname(script_path)
        # Get the path to the app bundle's "Resources" directory
        resources_dir = os.path.join(script_dir)
        wd = resources_dir#'/Users/zhihuang/Desktop/enable_nuclei.io/app/build_install_macOS/dist/main.app/Contents/Resources/'
    else:
        # Windows
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

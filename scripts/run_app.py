"""
Author: Martin Zůbek
Date: 2026-04-10
File: run_app.py
Description: This script is responsible for running the application.
"""

import sys

from PyQt5.QtWidgets import QApplication

from src.gui.app_window import AppWindow

if __name__ == "__main__":
    # Create the application
    application: QApplication = QApplication(sys.argv)
    # Create and show the main window
    app_window: AppWindow = AppWindow()
    app_window.show()
    sys.exit(application.exec())

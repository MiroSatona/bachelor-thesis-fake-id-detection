"""
Author: Martin Zůbek
Date: 2026-04-10
File: classify_runner.py
Description: Background QThread for running model classification without blocking the UI.
"""

from pathlib import Path

from PyQt5.QtCore import QThread, pyqtSignal

from src.model.app_model import AppModel


class ClassifyRunner(QThread):
    """
    Class which is responsible for running the model classification in a separate thread.
    
    Class attributes:
        finished_signal (pyqtSignal): Signal emitted when classification is finished.
        error_signal (pyqtSignal): Signal emitted when an error occurs during classification.
    
    Attributes:
        model (AppModel): The model for the classification.
        data_path (Path): The path to the data to classify.
    """

    finished_signal: pyqtSignal = pyqtSignal(bool, object)
    error_signal: pyqtSignal = pyqtSignal(str)

    def __init__(self, model: AppModel, data_path: Path) -> None:
        """
        Constructor for ClassifyRunner.
        
        Arguments:
            model (AppModel): The model for the classification.
            data_path (Path): The path to the data to classify.

        Returns:
            None
        """
        super().__init__()
        self.model = model
        self.data_path = data_path

    def run(self) -> None:
        """
        Method that runs the classification in a separate thread. It emits a signal with the 
        result when finished or an error message if an exception occurs.

        Arguments:
            None
        
        Returns:
            None
        """
        try:
            classify_result, fraud_map = self.model.classify(self.data_path)
            self.finished_signal.emit(classify_result, fraud_map)
        except Exception as classify_exception:
            self.error_signal.emit(str(classify_exception))

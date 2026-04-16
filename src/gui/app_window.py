"""
Author: Martin Zůbek
Date: 2026-04-10
File: app_window.py
Description: File containing the implementation of the AppWindow class which is responsible
for creating GUI and calling the methods for classifying identity documents.
"""


from pathlib import Path

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon, QPixmap
from PyQt5.QtWidgets import (
    QFileDialog,
    QLabel,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from src.data_processing.constants import IMAGE_EXTENSIONS
from src.model.app_model import AppModel


class AppWindow(QMainWindow):
    """
    Class representing the main window of the application. It is responsible for creating the GUI 
    and calling the methods for classifying identity documents.

    Attributes:
        model (AppModel): The model for classifying the identity documents.
        data_path (Path | None): Path to the selected image for classification.
        title_label (QLabel): Title of the application.
        image_handler (QPushButton): Handler of the image of the identity document.
        image_description_label (QLabel): Description of the state of the image selection.
        result_label (QLabel): Label for displaying the result of the classification.
        classify_button (QPushButton): Button for classifying the image of the identity document.
    """
    def __init__(self) -> None:
        """
        Constructor of the AppWindow class. It initializes the model for classifying the identity
        documents, creates the GUI elements, and connects them to the methods. 
        
        Arguments:
            None

        Returns:
            None
        """
        super().__init__()
        # Initialize the model for classifying the identity documents
        self.model: AppModel = AppModel()
        # Path to the selected image for classification
        self.data_path: Path | None = None

        # Title and size of the window
        self.setWindowTitle("Application for detecting generated fake ID cards")
        self.setMinimumSize(700, 600)

        # Title of the application
        self.title_label: QLabel = QLabel("Classification of identity documents")
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Handler of the image of the identity document
        self.image_handler: QPushButton = QPushButton("Click to upload ID card image")
        self.image_handler.setCursor(Qt.CursorShape.PointingHandCursor)
        self.image_handler.setFixedHeight(320)

        # Description of the state of the image selection
        self.image_description_label: QLabel = QLabel("No image selected")
        self.image_description_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Label for displaying the result of the classification
        self.result_label: QLabel = QLabel("Result: -")
        self.result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Button for classifying the image of the identity document
        self.classify_button: QPushButton = QPushButton("Classify")
        self.classify_button.setEnabled(False)

        # Connect the identity image handler to the method
        self.image_handler.clicked.connect(self.select_identity_document)
        # Connect the classify button to the method for classifying the identity document
        self.classify_button.clicked.connect(self.classify_identity_document)

        # Create the layout of the window
        layout: QVBoxLayout = QVBoxLayout()
        layout.setSpacing(16)
        layout.setContentsMargins(30, 30, 30, 30)

        # Add the elemnts to the layout
        layout.addWidget(self.title_label)
        layout.addWidget(self.image_handler)
        layout.addWidget(self.image_description_label)
        layout.addWidget(self.result_label)
        layout.addWidget(self.classify_button)
        
        # Set the layout on a container widget and use it as the central widget
        container: QWidget = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        # Set the style of the application
        self.style()

    def style(self) -> None:
        """
        Method for setting the style of the elements in the application.
        
        Arguments:
            None

        Returns:
            None
        """
        # Set the style of the initial elements in the application
        self.setStyleSheet(
            """
            QMainWindow {
                background-color: #f5f7fb;
            }

            QLabel {
                color: #1f2937;
                font-size: 15px;
            }

            QLabel#title {
                font-size: 24px;
                font-weight: 700;
                color: #111827;
            }

            QPushButton {
                background-color: #2563eb;
                color: white;
                border: none;
                border-radius: 10px;
                padding: 12px 18px;
                font-size: 15px;
                font-weight: 600;
            }

            QPushButton:hover {
                background-color: #1d4ed8;
            }

            QPushButton:disabled {
                background-color: #9ca3af;
                color: #f3f4f6;
            }
            """
        )

        # Set the object name for the title label
        self.title_label.setObjectName("title")

        # Set the style of the image handler
        self.image_handler.setStyleSheet(
            """
            QPushButton {
                background-color: #e5e7eb;
                color: #374151;
                border: 2px dashed #9ca3af;
                border-radius: 14px;
                font-size: 16px;
                font-weight: 500;
                text-align: center;
                padding: 20px;
            }

            QPushButton:hover {
                background-color: #dbe3ee;
                border: 2px dashed #6b7280;
            }
            """
        )

        # Set the style of the result label
        self.result_label.setStyleSheet(
            """
            QLabel {
                background-color: white;
                border: 1px solid #d1d5db;
                border-radius: 10px;
                padding: 12px;
                font-size: 16px;
                font-weight: 600;
            }
            """
        )

    def select_identity_document(self) -> None:
        """
        Method for selecting the image of the identity document. By QFileDialog getting the path to
        the image and displaying in the application.
        
        Arguments:
            None
            
        Returns:
            None
        """
        # Get path like string to the image of the identity document
        data_path_str: str = QFileDialog.getOpenFileName(
            self,
            "Select identity document image",
            "",
            f"Image files ({' '.join(f'*{extension}' for extension in IMAGE_EXTENSIONS)})"
        )[0]

        # Check if the identity document was provided
        if not data_path_str:
            return

        # Create the Path object from the string
        self.data_path = Path(data_path_str)
        # Update the description of the selected identity document image
        self.image_description_label.setText(f"Selected image: {self.data_path.name}")
        # Reset the result label
        self.result_label.setText("Result: -")
        # Enable the classify button
        self.classify_button.setEnabled(True)

        # Load the image of the identity document
        identity_document_pixmap: QPixmap = QPixmap(str(self.data_path))
        # If the image is loaded successfully, display it in the image handler
        if not identity_document_pixmap.isNull():
            scaled_pixmap = identity_document_pixmap.scaled(
                self.image_handler.width() - 20,
                self.image_handler.height() - 20,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            self.image_handler.setIconSize(scaled_pixmap.size())
            self.image_handler.setIcon(QIcon(scaled_pixmap))
            self.image_handler.setText("")
        else:
            self.image_handler.setText("Cannot preview the selected identity document image")

    def classify_identity_document(self) -> None:
        """
        Method for handling the classification of the identity document. It calls the method 
        for classifying the identity document in the model and updates the result label 
        with the result of the classification.
        
        Arguments:
            None
            
        Returns:
            None
        """

        # Check if the identity document image is selected
        if self.data_path is None:
            self.result_label.setText("Result: No data loaded")
            return

        # Classify the identity document and update the result label
        try:
            # Classify the identity document and get the result
            fraud_document: bool = self.model.classify(self.data_path)

            # Fraud identity document
            if fraud_document:
                self.result_label.setText("Result: The ID card is fake")
                self.result_label.setStyleSheet(
                    """
                    QLabel {
                        background-color: #fef2f2;
                        color: #991b1b;
                        border: 1px solid #fecaca;
                        border-radius: 10px;
                        padding: 12px;
                        font-size: 16px;
                        font-weight: 600;
                    }
                    """
                )
            # Genuine identity document
            else:
                self.result_label.setText("Result: The ID card is genuine")
                self.result_label.setStyleSheet(
                    """
                    QLabel {
                        background-color: #ecfdf5;
                        color: #065f46;
                        border: 1px solid #a7f3d0;
                        border-radius: 10px;
                        padding: 12px;
                        font-size: 16px;
                        font-weight: 600;
                    }
                    """
                )
        # Exception during the classification
        except Exception as exception:
            self.result_label.setText(f"Result: Error during classification: {str(exception)}")
            self.result_label.setStyleSheet(
                """
                QLabel {
                    background-color: #fff7ed;
                    color: #9a3412;
                    border: 1px solid #fdba74;
                    border-radius: 10px;
                    padding: 12px;
                    font-size: 16px;
                    font-weight: 600;
                }
                """
            )
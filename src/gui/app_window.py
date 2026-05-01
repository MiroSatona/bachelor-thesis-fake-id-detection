"""
Author: Martin Zůbek
Date: 2026-04-10
File: app_window.py
Description: File containing the implementation of the AppWindow class which is responsible
for creating GUI and calling the methods for classifying identity documents.
"""


from pathlib import Path

import numpy
from PIL import Image, ImageDraw, ImageFont
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon, QImage, QPixmap, QResizeEvent
from PyQt5.QtWidgets import (
    QFileDialog,
    QLabel,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from src.data_processing.constants import IMAGE_EXTENSIONS
from src.gui.classify_runner import ClassifyRunner
from src.gui.constants import (
    FRAUD_MAP_HEADER_BACKGROUND,
    FRAUD_MAP_HEADER_FONT_PATH,
    FRAUD_MAP_HEADER_FONT_SIZE,
    FRAUD_MAP_HEADER_HEIGHT,
    FRAUD_MAP_OVERLAY_BLUE,
    FRAUD_MAP_OVERLAY_GREEN,
    FRAUD_MAP_OVERLAY_MAX_ALPHA,
    FRAUD_MAP_OVERLAY_RED,
    LAYOUT_MARGIN,
    LAYOUT_SPACING,
    MAX_PREVIEW_HEIGHT,
    PREVIEW_HORIZONTAL_MARGIN,
    TITLE_FONT_MAX_SIZE,
    TITLE_FONT_MIN_SIZE,
    TITLE_FONT_WIDTH_DIVISOR,
    WINDOW_MIN_HEIGHT,
    WINDOW_MIN_WIDTH,
)
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
        # Window pixmap for responsive rescaling
        self.window_pixmap: QPixmap | None = None
        # Runner for the classification in a background thread
        self.classify_runner: ClassifyRunner | None = None
        # Fraud map preview 
        self.fraud_map: Image.Image | None = None

        # Title and size of the window
        self.setWindowTitle("Application for detecting generated fake ID cards")
        self.setMinimumSize(WINDOW_MIN_WIDTH, WINDOW_MIN_HEIGHT)

        # Title of the application
        self.title_label: QLabel = QLabel("Classification of identity documents")
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Handler of the image of the identity document
        self.image_handler: QPushButton = QPushButton("Click to upload ID card image")
        self.image_handler.setCursor(Qt.CursorShape.PointingHandCursor)
        self.image_handler.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

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
        layout.setSpacing(LAYOUT_SPACING)
        layout.setContentsMargins(LAYOUT_MARGIN, LAYOUT_MARGIN, LAYOUT_MARGIN, LAYOUT_MARGIN)

        # Add the elements to the layout
        layout.addWidget(self.title_label)
        layout.addWidget(self.image_handler, stretch=1)
        layout.addWidget(self.image_description_label)
        layout.addWidget(self.result_label)
        layout.addWidget(self.classify_button)

        # Set the layout on a container widget
        container: QWidget = QWidget()
        container.setLayout(layout)

        # Add the container widget to a scroll area and set it
        # as the central widget of the window
        scroll_area: QScrollArea = QScrollArea()
        scroll_area.setWidget(container)
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameStyle(0)
        self.setCentralWidget(scroll_area)

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

    def resizeEvent(self, event: QResizeEvent | None) -> None:
        """
        Overriden method for handling the resize event of the window.
        
        Arguments:
            event (QResizeEvent): The resize event.
        
        Returns:
            None
        """
        super().resizeEvent(event)
        self.resize_title_font()
        self.resize_identity_document_preview()
        self.resize_fraud_map_preview()

    def resize_title_font(self) -> None:
        """
        Method for resizing the font-size of the title of the application.

        Arguments:
            None
        Returns:
            None
        """
        # Calculate the font size based on the width of the window, 
        # with a minimum of 18 and a maximum of 48
        font_size: int = max(
            TITLE_FONT_MIN_SIZE, min(TITLE_FONT_MAX_SIZE, self.width() // TITLE_FONT_WIDTH_DIVISOR)
        )
        self.title_label.setStyleSheet(
            f"""
            QLabel {{
                font-size: {font_size}px;
                font-weight: 700;
                color: #111827;
            }}
            """
        )

    def resize_identity_document_preview(self) -> None:
        """
        Method for resizing the preview of the identity document image
        when the window is resized.
        
        Arguments:
            None
        Returns:
            None
        """
        # Check if the window pixmap is loaded
        if self.window_pixmap is None or self.window_pixmap.isNull():
            return
        
        # Scale the preview to fit the available viewport width and a fixed max height,
        # capped at the original pixmap size to prevent upscaling
        scroll_area = self.centralWidget()
        if not isinstance(scroll_area, QScrollArea):
            return
        viewport = scroll_area.viewport()
        if viewport is None:
            return
        available_width: int = max(1, viewport.width() - PREVIEW_HORIZONTAL_MARGIN)
        target_width: int = min(available_width, self.window_pixmap.width())
        target_height: int = min(MAX_PREVIEW_HEIGHT, self.window_pixmap.height())
        scaled: QPixmap = self.window_pixmap.scaled(
            target_width,
            target_height,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.image_handler.setIconSize(scaled.size())
        self.image_handler.setIcon(QIcon(scaled))

    def resize_fraud_map_preview(self) -> None:
        """
        Method for resizing the preview of the fraud map image when
        the window is resized.

        Arguments:
            None
        Returns:
            None
        """
        # Check if the fraud map composite is stored
        if self.fraud_map is None:
            return

        # Get the the available width of area minus the margins
        scroll_area = self.centralWidget()
        if not isinstance(scroll_area, QScrollArea):
            return
        viewport = scroll_area.viewport()
        if viewport is None:
            return
        available_width: int = max(1, viewport.width() - PREVIEW_HORIZONTAL_MARGIN)

        # Scale to fit the available width
        composite_width, composite_height = self.fraud_map.size
        scale: float = min(1.0, available_width / composite_width)
        scaled_width: int = max(1, int(composite_width * scale))
        scaled_height: int = max(1, int(composite_height * scale))

        # Scale the stored composite to the current available width
        scaled_composite: Image.Image = self.fraud_map.resize(
            (scaled_width, scaled_height), Image.Resampling.LANCZOS
        )

        # Create a new image with space for the text header above the document image
        preview_image: Image.Image = Image.new(
            "RGB",
            (scaled_width, FRAUD_MAP_HEADER_HEIGHT + scaled_height),
            FRAUD_MAP_HEADER_BACKGROUND,
        )

        # Load the font for the header text of the result
        font: ImageFont.FreeTypeFont | ImageFont.ImageFont
        try:
            font = ImageFont.truetype(
                FRAUD_MAP_HEADER_FONT_PATH, FRAUD_MAP_HEADER_FONT_SIZE
            )
        except (IOError, OSError):
            font = ImageFont.load_default()

        # Draw the result text centred in the header area
        draw: ImageDraw.ImageDraw = ImageDraw.Draw(preview_image)
        draw.text(
            (scaled_width // 2, FRAUD_MAP_HEADER_HEIGHT // 2),
            "Result: The ID card is fake",
            fill=(153, 27, 27),
            font=font,
            anchor="mm",
        )

        # Paste the scaled identity document image
        preview_image.paste(scaled_composite, (0, FRAUD_MAP_HEADER_HEIGHT))

        # Convert the PIL image to a QPixmap
        image_array = numpy.ascontiguousarray(numpy.asarray(preview_image, dtype=numpy.uint8))
        pixmap: QPixmap = QPixmap.fromImage(
            QImage(
                image_array.tobytes(),
                scaled_width,
                FRAUD_MAP_HEADER_HEIGHT + scaled_height,
                3 * scaled_width,
                QImage.Format.Format_RGB888,
            )
        )

        # Show the resulting pixmap in the result label
        self.result_label.setPixmap(pixmap)

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
        # Reset the result label text and style
        self.result_label.setText("Result: -")
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
        # Reset the fraud map when a new image is selected
        self.fraud_map = None

        # Enable the classify button
        self.classify_button.setEnabled(True)

        # Load and store the original pixmap for responsive resizing
        self.window_pixmap = QPixmap(str(self.data_path))
        if not self.window_pixmap.isNull():
            self.resize_identity_document_preview()
            self.image_handler.setText("")
        else:
            self.window_pixmap = None
            self.image_handler.setText(
                "Cannot preview the selected identity document image"
            )

    def classify_identity_document(self) -> None:
        """
        Method for handling the classification of the identity document.

        Arguments:
            None

        Returns:
            None
        """

        # Check if the identity document image is selected
        if self.data_path is None:
            self.result_label.setText("Result: No data loaded")
            return

        # Show information about the classifying and disable 
        # the classify button and image handler while classifying
        self.result_label.setText("Result: Classifying...")
        self.result_label.setStyleSheet(
            """
            QLabel {
                background-color: #eff6ff;
                color: #1e40af;
                border: 1px solid #bfdbfe;
                border-radius: 10px;
                padding: 12px;
                font-size: 16px;
                font-weight: 600;
            }
            """
        )
        self.classify_button.setEnabled(False)
        self.image_handler.setEnabled(False)
        self.classify_runner = ClassifyRunner(self.model, self.data_path)
        self.classify_runner.finished_signal.connect(self.show_classification_result)
        self.classify_runner.error_signal.connect(self.handle_classify_error)
        self.classify_runner.start()

    def show_classification_result(
        self, classification_result: bool, fraud_map: numpy.ndarray
    ) -> None:
        """
        Method for showing the result  in the result label after the classification is finished.
        
        Arguments:
            classification_result (bool): The result of the classification, 
            True if the ID card is fake, False if it is genuine.
            fraud_map (object): Per-pixel manipulation fraud map as numpy.ndarray (H, W) in [0, 1],
            or None when not available.

        Returns:
            None    
        """
        # Enable the classify button and image handler after classification
        self.classify_button.setEnabled(True)
        self.image_handler.setEnabled(True)
        if classification_result:
            # Show the result of the classification for the fake ID card and show the fraud map
            if fraud_map is not None:
                self.show_fraud_map(fraud_map)
            else:
                self.fraud_map = None
                self.result_label.setPixmap(QPixmap())
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
        else:
            # Show the result of the classification for the genuine ID card
            self.fraud_map = None
            self.result_label.setPixmap(QPixmap())
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

    def handle_classify_error(self, exception_message: str) -> None:
        """
        Method for handling the error during the classification and showing the error message 
        in the result label.
        
        Arguments:
            exception_message (str): The error message to be displayed.

        Returns:
            None
        """
        self.classify_button.setEnabled(True)
        self.image_handler.setEnabled(True)
        self.result_label.setText(f"Result: Error during classification: {exception_message}")
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

    def show_fraud_map(self, fraud_map: numpy.ndarray) -> None:
        """
        Method for create the red overlay of the fraud map on the original identity document image
        and showing it in the result label.

        Arguments:
            fraud_map (numpy.ndarray): Per-pixel manipulation fraud map as numpy.ndarray.

        Returns:
            None
        """

        # Check if the data path was provided
        if self.data_path is None:
            return

        # Open the original identity document image and convert it to RGBA
        # for having the alpha channel for the red overlay
        identity_document_image: Image.Image =(
            Image.open(str(self.data_path)).convert("RGBA")
        )
        # Get the width and height of the original identity document image
        identity_document_width, identity_document_height = identity_document_image.size

        # Clipped fraud map
        clipped_fraud_map: numpy.ndarray = numpy.clip(
            numpy.asarray(fraud_map, dtype=numpy.float32), 0.0, 1.0
        )

        # Convert the clipped fraud map to a grayscale PIL image, resize it to the document
        # dimensions using bilinear interpolation, then convert back to a numpy array in [0, 1]
        fraud_map_pil: Image.Image = Image.fromarray(
            (clipped_fraud_map * 255).astype(numpy.uint8), mode="L"
        ).resize((identity_document_width, identity_document_height), Image.Resampling.BILINEAR)
        fraud_map_normalized: numpy.ndarray = (
            numpy.asarray(fraud_map_pil, dtype=numpy.float32) / 255.0
        )

        # Create the red overlay
        red_overlay: numpy.ndarray = numpy.zeros(
            (identity_document_height, identity_document_width, 4), dtype=numpy.uint8
        )
        red_overlay[:, :, 0] = FRAUD_MAP_OVERLAY_RED
        red_overlay[:, :, 1] = FRAUD_MAP_OVERLAY_GREEN
        red_overlay[:, :, 2] = FRAUD_MAP_OVERLAY_BLUE
        red_overlay[:, :, 3] = (
            (fraud_map_normalized * FRAUD_MAP_OVERLAY_MAX_ALPHA).astype(numpy.uint8)
        )

        # Merge the original identity document image with the red overlay
        self.fraud_map = Image.alpha_composite(
            identity_document_image, Image.fromarray(red_overlay, mode="RGBA")
        ).convert("RGB")

        # Set the the result label to have fake result
        self.result_label.setStyleSheet(
            """
            QLabel {
                background-color: #fef2f2;
                border: 1px solid #fecaca;
                border-radius: 10px;
                padding: 8px;
            }
            """
        )

        self.resize_fraud_map_preview()

"""Install appropriate packages if not already installed"""

import sys
import os
import time
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QListWidget, QSlider, QSpinBox,
    QGroupBox, QMessageBox, QTextEdit, QLineEdit, QFormLayout, QTabWidget,
    QProgressBar, QListWidgetItem, QDialog, QScrollArea, QTableWidget, QTableWidgetItem, QHeaderView,
    QCheckBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QPixmap, QPainter, QPen, QColor, QImage, QBrush
from PIL import Image
import tifffile
from skimage import restoration, color, measure, exposure
from scipy import ndimage, stats
from matplotlib.path import Path as mplPath
import cv2
import glob


def load_tiff_image(filepath):
    """Load TIFF image using PIL to handle all compression types"""
    img = Image.open(filepath)
    # Convert to numpy array
    img_array = np.array(img)
    return img_array


def ensure_grayscale(img):
    """Convert image to grayscale if needed"""
    if img is None:
        return None
    if img.ndim == 3:
        # Handle RGBA (4 channels) by dropping alpha channel
        if img.shape[2] == 4:
            img = img[:, :, :3]  # Keep only RGB channels
        # Convert RGB to grayscale
        img = (color.rgb2gray(img) * 255).astype(img.dtype)
    if img.ndim > 2:
        img = img.squeeze()
    return img


class MorphologyCalculator:
    """
    Calculate morphological parameters for microglia.

    Calculates 6 basic metrics: roundness, eccentricity, soma area,
    mask area, perimeter, and cell spread.
    """

    def __init__(self, image, pixel_size_um, use_imagej=False):
        self.image = image
        self.pixel_size = pixel_size_um
        # Note: use_imagej parameter kept for backwards compatibility but not used

    def calculate_all_parameters(self, cell_mask, soma_centroid, soma_area_um2=None):
        """Calculate ONLY simple parameters - Sholl, Fractal, Hull, Skeleton done in ImageJ"""
        params = {}
        # ✅ ONLY: Simple descriptors (fast Python metrics)
        params.update(self._calculate_simple_descriptors(cell_mask, soma_area_um2))

        return params

    def _calculate_simple_descriptors(self, mask, soma_area_um2=None):
        params = {}
        props = measure.regionprops(mask.astype(int))[0] if np.any(mask) else None
        if props:
            params['perimeter'] = props.perimeter * self.pixel_size
            params['mask_area'] = props.area * (self.pixel_size ** 2)

            major_axis = props.major_axis_length
            minor_axis = props.minor_axis_length

            if minor_axis > 0:
                params['eccentricity'] = major_axis / minor_axis
            else:
                params['eccentricity'] = 0

            if major_axis > 0:
                params['roundness'] = minor_axis / major_axis
            else:
                params['roundness'] = 0

            centroid = np.array(props.centroid)
            coords = np.array(props.coords)

            top_point = coords[coords[:, 0].argmin()]
            bottom_point = coords[coords[:, 0].argmax()]
            left_point = coords[coords[:, 1].argmin()]
            right_point = coords[coords[:, 1].argmax()]

            extremities = np.array([top_point, bottom_point, left_point, right_point])
            distances = np.sqrt(np.sum((extremities - centroid) ** 2, axis=1))
            params['cell_spread'] = np.mean(distances) * self.pixel_size

            if soma_area_um2 is not None:
                params['soma_area'] = soma_area_um2
            else:
                params['soma_area'] = props.area * 0.1 * (self.pixel_size ** 2)
        else:
            params = {k: 0 for k in ['perimeter', 'mask_area', 'eccentricity',
                                     'roundness', 'cell_spread', 'soma_area']}
        return params


class BatchProcessingThread(QThread):
    """Thread for batch processing images in the background"""
    progress = pyqtSignal(int)
    status_update = pyqtSignal(str)
    finished_image = pyqtSignal(str, str, object)
    error_occurred = pyqtSignal(str)

    def __init__(self, image_data_list, output_dir):
        super().__init__()
        self.image_data_list = image_data_list
        self.output_dir = output_dir

    def run(self):
        try:
            total = len(self.image_data_list)
            for i, (
                    img_path, img_name, radius, denoise_enabled, denoise_size, sharpen_enabled,
                    sharpen_amount) in enumerate(
                self.image_data_list):
                try:
                    self.status_update.emit(f"Processing: {img_name}")
                    img = load_tiff_image(img_path)
                    img = ensure_grayscale(img)
                    img_dtype = img.dtype
                    background = restoration.rolling_ball(img, radius=radius)
                    result = img - background
                    result = np.clip(result, 0, np.iinfo(img_dtype).max)

                    # Apply optional denoising
                    if denoise_enabled:
                        result = ndimage.median_filter(result, size=denoise_size)

                    # Apply optional sharpening
                    if sharpen_enabled:
                        blurred = ndimage.gaussian_filter(result.astype(np.float32), sigma=2)
                        result_float = result.astype(np.float32)
                        sharpened = result_float + sharpen_amount * (result_float - blurred)
                        result = np.clip(sharpened, 0, np.iinfo(img_dtype).max).astype(img_dtype)

                    name = os.path.splitext(img_name)[0]
                    out_path = os.path.join(self.output_dir, f"{name}_processed.tif")
                    tifffile.imwrite(out_path, result.astype(img_dtype))
                    self.finished_image.emit(out_path, img_name, result)
                    self.progress.emit(int((i + 1) / total * 100))
                except Exception as e:
                    self.error_occurred.emit(f"Error: {img_name}: {e}")
        except Exception as e:
            self.error_occurred.emit(f"Fatal error: {e}")


class MorphologyCalculationThread(QThread):
    """Thread for calculating morphology parameters in the background"""
    progress = pyqtSignal(int, str)  # progress percentage, status message
    finished = pyqtSignal(list)  # list of results
    error_occurred = pyqtSignal(str)

    def __init__(self, approved_masks, pixel_size, use_imagej, images):
        super().__init__()
        self.approved_masks = approved_masks
        self.pixel_size = pixel_size
        self.use_imagej = use_imagej
        self.images = images

    def run(self):
        import sys
        import os

        # Suppress Java warnings during processing to speed up
        old_stderr = sys.stderr
        sys.stderr = open(os.devnull, 'w')

        try:
            all_results = []
            total = len(self.approved_masks)

            for i, flat_data in enumerate(self.approved_masks):
                mask_data = flat_data['mask_data']
                processed_img = flat_data['processed_img']
                img_name = flat_data['image_name']

                self.progress.emit(
                    int((i + 1) / total * 100),
                    f"Processing {mask_data['soma_id']} ({i + 1}/{total})"
                )

                # Find soma centroid and area from original image data
                img_data = self.images[img_name]
                soma_centroid = img_data['somas'][mask_data['soma_idx']]
                soma_area_um2 = mask_data.get('soma_area_um2', None)

                # Time tracking for diagnostics
                import time
                start = time.time()

                calculator = MorphologyCalculator(processed_img, self.pixel_size, use_imagej=self.use_imagej)
                params = calculator.calculate_all_parameters(mask_data['mask'], soma_centroid, soma_area_um2)

                elapsed = time.time() - start
                # Temporarily restore stderr to print timing
                sys.stderr = old_stderr
                print(f"  ✓ {mask_data['soma_id']}: {elapsed:.1f}s")
                sys.stderr = open(os.devnull, 'w')

                params['image_name'] = os.path.splitext(img_name)[0]
                params['soma_id'] = mask_data['soma_id']
                params['soma_idx'] = mask_data['soma_idx']
                params['area_um2'] = mask_data['area_um2']

                all_results.append(params)

            self.finished.emit(all_results)

        except Exception as e:
            self.error_occurred.emit(str(e))
        finally:
            # Always restore stderr
            if sys.stderr != old_stderr:
                sys.stderr.close()
                sys.stderr = old_stderr


class BackgroundRemovalThread(QThread):
    progress = pyqtSignal(int)
    status_update = pyqtSignal(str)
    finished_image = pyqtSignal(str, str, object)
    error_occurred = pyqtSignal(str)

    def __init__(self, image_data_list, output_dir):
        super().__init__()
        self.image_data_list = image_data_list
        self.output_dir = output_dir

    def run(self):
        try:
            total = len(self.image_data_list)
            for i, (
                    img_path, img_name, radius, denoise_enabled, denoise_size, sharpen_enabled,
                    sharpen_amount) in enumerate(
                self.image_data_list):
                try:
                    self.status_update.emit(f"Processing: {img_name}")
                    img = load_tiff_image(img_path)
                    img = ensure_grayscale(img)
                    img_dtype = img.dtype
                    background = restoration.rolling_ball(img, radius=radius)
                    result = img - background
                    result = np.clip(result, 0, np.iinfo(img_dtype).max)

                    # Apply optional denoising
                    if denoise_enabled:
                        result = ndimage.median_filter(result, size=denoise_size)

                    # Apply optional sharpening
                    if sharpen_enabled:
                        blurred = ndimage.gaussian_filter(result.astype(np.float32), sigma=2)
                        result_float = result.astype(np.float32)
                        sharpened = result_float + sharpen_amount * (result_float - blurred)
                        result = np.clip(sharpened, 0, np.iinfo(img_dtype).max).astype(img_dtype)

                    name = os.path.splitext(img_name)[0]
                    out_path = os.path.join(self.output_dir, f"{name}_processed.tif")
                    tifffile.imwrite(out_path, result.astype(img_dtype))
                    self.finished_image.emit(out_path, img_name, result)
                    self.progress.emit(int((i + 1) / total * 100))
                except Exception as e:
                    self.error_occurred.emit(f"Error: {img_name}: {e}")
        except Exception as e:
            self.error_occurred.emit(f"Fatal error: {e}")


class InteractiveImageLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_widget = parent
        self.pix_source = None
        self.centroids = []
        self.mask_overlay = None
        self.polygon_pts = []
        self.soma_mode = False
        self.polygon_mode = False
        self.setMinimumSize(400, 400)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("border: 2px solid #cccccc; background-color: #f5f5f5;")
        # Allow label to expand
        from PyQt5.QtWidgets import QSizePolicy
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setScaledContents(False)

        # Zoom and pan variables
        self.zoom_factor = 1.0
        self.pan_offset = [0, 0]
        self.is_panning = False
        self.last_pan_point = None
        self.base_pixmap = None

    def set_image(self, qpix, centroids=None, mask_overlay=None, polygon_pts=None):
        """Set image and overlays - does NOT apply zoom yet"""
        self.base_pixmap = qpix
        self.pix_source = qpix
        self.centroids = centroids or []
        self.mask_overlay = mask_overlay
        self.polygon_pts = polygon_pts or []
        self._update_display()

    def _update_display(self):
        """Update the displayed image with current zoom"""
        if not self.base_pixmap:
            super().setPixmap(QPixmap())
            return

        label_size = self.size()

        # First fit to label size
        fitted_pix = self.base_pixmap.scaled(
            label_size.width(),
            label_size.height(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )

        # Then apply zoom if needed
        if self.zoom_factor != 1.0:
            scaled_w = int(fitted_pix.width() * self.zoom_factor)
            scaled_h = int(fitted_pix.height() * self.zoom_factor)
            fitted_pix = fitted_pix.scaled(
                scaled_w, scaled_h,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )

        super().setPixmap(fitted_pix)
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        if not self.pix_source:
            return
        painter = QPainter(self)
        if self.mask_overlay is not None:
            self._draw_mask_overlay(painter)
        if self.centroids:
            pen = QPen(QColor(255, 0, 0), 3)
            painter.setPen(pen)
            for centroid in self.centroids:
                x, y = self._to_display_coords(centroid)
                painter.drawEllipse(int(x - 6), int(y - 6), 12, 12)
        if self.polygon_mode and len(self.polygon_pts) > 0:
            self._draw_polygon(painter)
        painter.end()

    def resizeEvent(self, event):
        """Re-scale the image when the label is resized"""
        super().resizeEvent(event)
        if self.base_pixmap is not None:
            self._update_display()

    def auto_zoom_to_point(self, point, zoom_level=3.0):
        """Auto zoom and center on a specific point"""
        if not self.base_pixmap or not point:
            return

        self.zoom_factor = zoom_level
        img_y, img_x = point

        # Get original image dimensions
        orig_w = self.base_pixmap.width()
        orig_h = self.base_pixmap.height()

        # Get label dimensions
        label_w = self.size().width()
        label_h = self.size().height()

        # Calculate fitted size (before zoom)
        fitted_pix = self.base_pixmap.scaled(
            label_w, label_h,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        fitted_w = fitted_pix.width()
        fitted_h = fitted_pix.height()

        # Calculate zoomed size
        zoomed_w = fitted_w * self.zoom_factor
        zoomed_h = fitted_h * self.zoom_factor

        # Calculate where the point is in the fitted/zoomed image
        scale_x = fitted_w / orig_w
        scale_y = fitted_h / orig_h

        point_x_in_fitted = img_x * scale_x
        point_y_in_fitted = img_y * scale_y

        # Apply zoom to point position
        point_x_in_zoomed = point_x_in_fitted * self.zoom_factor
        point_y_in_zoomed = point_y_in_fitted * self.zoom_factor

        # Calculate pan offset to center the point
        self.pan_offset[0] = (label_w / 2) - point_x_in_zoomed
        self.pan_offset[1] = (label_h / 2) - point_y_in_zoomed

        self._update_display()

        if self.parent_widget and hasattr(self.parent_widget, 'log'):
            self.parent_widget.log(f"✓ Zoomed to {zoom_level}x at point ({img_y}, {img_x})")

    def reset_view(self):
        """Reset zoom and pan to default"""
        self.zoom_factor = 1.0
        self.pan_offset = [0, 0]
        self._update_display()

        if self.parent_widget and hasattr(self.parent_widget, 'log'):
            self.parent_widget.log("✓ Reset zoom to 1x")

    def _draw_mask_overlay(self, painter):
        """Draw mask overlay accounting for zoom and pan"""
        if self.mask_overlay is None:
            return
        mask = self.mask_overlay
        if not self.pix_source:
            return
        mask_coords = np.argwhere(mask > 0)
        if len(mask_coords) == 0:
            return

        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(0, 255, 0))
        painter.setOpacity(0.4)

        img_h, img_w = self.pix_source.height(), self.pix_source.width()
        label_w, label_h = self.size().width(), self.size().height()

        # Get fitted size
        if not self.base_pixmap:
            return

        fitted_pix = self.base_pixmap.scaled(
            label_w, label_h,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        fitted_w = fitted_pix.width()
        fitted_h = fitted_pix.height()

        scale_x = fitted_w / img_w
        scale_y = fitted_h / img_h

        # Account for zoom
        rect_w = max(1, int(scale_x * self.zoom_factor))
        rect_h = max(1, int(scale_y * self.zoom_factor))

        for coord in mask_coords:
            img_y, img_x = coord[0], coord[1]
            x_pos, y_pos = self._to_display_coords((img_y, img_x))
            painter.drawRect(int(x_pos), int(y_pos), rect_w, rect_h)

        painter.setOpacity(1.0)

    def _draw_polygon(self, painter):
        pen = QPen(QColor(255, 165, 0), 3)
        painter.setPen(pen)
        for i in range(len(self.polygon_pts)):
            p1 = self._to_display_coords(self.polygon_pts[i])
            p2 = self._to_display_coords(self.polygon_pts[(i + 1) % len(self.polygon_pts)])
            painter.drawLine(int(p1[0]), int(p1[1]), int(p2[0]), int(p2[1]))
        pen = QPen(QColor(0, 0, 255), 2)
        painter.setPen(pen)
        painter.setBrush(QColor(0, 0, 255))
        for pt in self.polygon_pts:
            x, y = self._to_display_coords(pt)
            painter.drawEllipse(int(x - 4), int(y - 4), 8, 8)

    def _to_display_coords(self, img_coords):
        """Convert image coordinates to display coordinates accounting for zoom and pan"""
        if not self.pix_source:
            return 0, 0

        img_y, img_x = img_coords[0], img_coords[1]
        img_h, img_w = self.pix_source.height(), self.pix_source.width()
        label_w, label_h = self.size().width(), self.size().height()

        # Get fitted size (before zoom)
        fitted_pix = self.base_pixmap.scaled(
            label_w, label_h,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        ) if self.base_pixmap else self.pix_source
        fitted_w = fitted_pix.width()
        fitted_h = fitted_pix.height()

        # Calculate scale from image to fitted
        scale_x = fitted_w / img_w
        scale_y = fitted_h / img_h

        # Apply scale and zoom
        x_in_fitted = img_x * scale_x
        y_in_fitted = img_y * scale_y

        x_in_zoomed = x_in_fitted * self.zoom_factor
        y_in_zoomed = y_in_fitted * self.zoom_factor

        # Center the fitted/zoomed image in label
        offset_x = (label_w - fitted_w * self.zoom_factor) / 2
        offset_y = (label_h - fitted_h * self.zoom_factor) / 2

        # Apply offsets (including pan)
        x = x_in_zoomed + offset_x + self.pan_offset[0]
        y = y_in_zoomed + offset_y + self.pan_offset[1]

        return x, y

    def _to_image_coords(self, display_x, display_y):
        """Convert display coordinates to image coordinates accounting for zoom and pan"""
        if not self.parent_widget or not hasattr(self.parent_widget, 'get_current_processed_image'):
            return None
        current_img = self.parent_widget.get_current_processed_image()
        if current_img is None:
            return None

        img_shape = current_img.shape
        img_h, img_w = img_shape[:2]
        label_w, label_h = self.size().width(), self.size().height()

        # Get fitted size (before zoom)
        if not self.base_pixmap:
            return None

        fitted_pix = self.base_pixmap.scaled(
            label_w, label_h,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        fitted_w = fitted_pix.width()
        fitted_h = fitted_pix.height()

        # Calculate offsets
        offset_x = (label_w - fitted_w * self.zoom_factor) / 2
        offset_y = (label_h - fitted_h * self.zoom_factor) / 2

        # Remove offsets (including pan)
        x_in_zoomed = display_x - offset_x - self.pan_offset[0]
        y_in_zoomed = display_y - offset_y - self.pan_offset[1]

        # Remove zoom
        x_in_fitted = x_in_zoomed / self.zoom_factor
        y_in_fitted = y_in_zoomed / self.zoom_factor

        # Calculate scale from fitted to image
        scale_x = img_w / fitted_w
        scale_y = img_h / fitted_h

        # Convert to image coords
        img_x = round(x_in_fitted * scale_x)
        img_y = round(y_in_fitted * scale_y)

        # Clamp to image bounds
        img_x = max(0, min(img_x, img_w - 1))
        img_y = max(0, min(img_y, img_h - 1))

        return (img_y, img_x)

    def mousePressEvent(self, event):
        coords = self._to_image_coords(event.pos().x(), event.pos().y())
        if not coords:
            return
        if self.soma_mode and self.parent_widget:
            self.parent_widget.add_soma(coords)
        elif self.polygon_mode and self.parent_widget:
            # Left click adds point, right click finishes
            if event.button() == Qt.LeftButton:
                self.parent_widget.add_polygon_point(coords)
            elif event.button() == Qt.RightButton:
                self.parent_widget.finish_polygon()

    def mouseDoubleClickEvent(self, event):
        if self.polygon_mode and self.parent_widget:
            self.parent_widget.finish_polygon()

    def wheelEvent(self, event):
        """Handle mouse wheel for zooming"""
        delta = event.angleDelta().y()

        if delta > 0:
            self.zoom_factor *= 1.15
        else:
            self.zoom_factor /= 1.15

        self.zoom_factor = max(0.5, min(self.zoom_factor, 10.0))

        self._update_display()

        if self.parent_widget:
            self.parent_widget.log(f"Zoom: {self.zoom_factor:.2f}x")


class MicrogliaAnalysisGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.images = {}
        self.current_image_name = None
        self.batch_mode = False
        self.soma_picking_queue = []
        self.outlining_queue = []
        self.polygon_points = []
        self.output_dir = None
        self.masks_dir = None
        self.pixel_size = 0.3
        self.default_rolling_ball_radius = 50
        self.all_masks_flat = []
        self.mask_qa_idx = 0
        self.mask_qa_active = False
        self.soma_mode = False  # Initialize soma_mode to prevent crashes
        # Initialize display adjustment values
        self.brightness_value = 0
        self.contrast_value = 0
        # Mask generation settings (defaults)
        self.use_min_intensity = True
        self.min_intensity_percent = 30
        self.use_imagej = False
        self.init_ui()

    def keyPressEvent(self, event):
        key = event.key()

        # Handle polygon outlining mode shortcuts
        if self.processed_label.polygon_mode:
            if key == Qt.Key_Z or key == Qt.Key_Backspace:
                # Undo last point
                self.undo_last_polygon_point()
                return
            elif key == Qt.Key_Escape:
                # Restart current outline
                self.restart_polygon()
                return
            elif key == Qt.Key_Return or key == Qt.Key_Enter:
                # Alternative way to finish polygon
                self.finish_polygon()
                return
            elif key == Qt.Key_Home or key == Qt.Key_0:
                # Reset zoom
                self.processed_label.reset_view()
                return

        # Handle soma picking mode shortcuts
        if self.processed_label.soma_mode:
            if key == Qt.Key_Return or key == Qt.Key_Enter:
                # Done picking somas for current image
                self.done_with_current()
                return
            elif key == Qt.Key_Escape:
                # Cancel soma picking
                if self.batch_mode and self.soma_picking_queue:
                    reply = QMessageBox.question(
                        self, 'Cancel Soma Picking',
                        'Cancel soma picking for remaining images?',
                        QMessageBox.Yes | QMessageBox.No
                    )
                    if reply == QMessageBox.Yes:
                        self.soma_picking_queue = []
                        self.batch_mode = False
                        self.processed_label.soma_mode = False
                        self.done_btn.setEnabled(False)
                        self.log("✗ Soma picking cancelled")
                return

        # Handle mask QA mode shortcuts
        if not self.mask_qa_active:
            super().keyPressEvent(event)
            return
        try:
            if key == Qt.Key_A:
                self.approve_current_mask()
            elif key == Qt.Key_R:
                self.reject_current_mask()
            elif key == Qt.Key_Left:
                self.prev_mask()
            elif key == Qt.Key_Right:
                self.next_mask()
            elif key == Qt.Key_Space:
                self.approve_current_mask()
            else:
                super().keyPressEvent(event)
        except Exception as e:
            self.log(f"ERROR handling keypress: {e}")

    def init_ui(self):
        self.setWindowTitle("Microglia Analysis - Multi-Image Batch Processing")
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        left_panel = self._create_left_panel()
        main_layout.addWidget(left_panel)
        right_panel = self._create_right_panel()
        main_layout.addWidget(right_panel, 1)

        # Try multiple approaches to ensure full screen
        # Get the screen geometry
        from PyQt5.QtWidgets import QDesktopWidget
        screen = QDesktopWidget().screenGeometry()
        self.setGeometry(screen)
        self.showMaximized()  # Also maximize to be sure

    def _create_left_panel(self):
        panel = QWidget()
        panel.setFixedWidth(450)
        layout = QVBoxLayout(panel)
        file_group = QGroupBox("1. File Selection")
        file_layout = QVBoxLayout()
        select_btn = QPushButton("Select Image Folder")
        select_btn.clicked.connect(self.select_folder)
        file_layout.addWidget(select_btn)
        output_btn = QPushButton("Select Output Folder")
        output_btn.clicked.connect(self.select_output)
        file_layout.addWidget(output_btn)
        self.file_list = QListWidget()
        self.file_list.itemClicked.connect(self.on_image_selected)
        self.file_list.itemChanged.connect(self.on_item_checkbox_changed)
        file_layout.addWidget(self.file_list)
        btn_layout = QHBoxLayout()
        select_all_btn = QPushButton("Select All")
        select_all_btn.clicked.connect(self.select_all_images)
        btn_layout.addWidget(select_all_btn)
        clear_all_btn = QPushButton("Clear All")
        clear_all_btn.clicked.connect(self.clear_all_images)
        btn_layout.addWidget(clear_all_btn)
        label_btn = QPushButton("Image Labeling")
        label_btn.clicked.connect(self.open_image_labeling)
        btn_layout.addWidget(label_btn)
        file_layout.addLayout(btn_layout)
        file_group.setLayout(file_layout)
        layout.addWidget(file_group)
        param_group = QGroupBox("2. Parameters")
        param_layout = QVBoxLayout()
        form_layout = QFormLayout()
        self.pixel_size_input = QLineEdit(str(self.pixel_size))
        form_layout.addRow("Pixel size (μm/px):", self.pixel_size_input)
        param_layout.addLayout(form_layout)

        self.use_imagej = False

        rb_layout = QHBoxLayout()
        # CLAHE (Contrast Limited Adaptive Histogram Equalization)
        self.clahe_check = QCheckBox("Apply CLAHE (Enhance Local Contrast)")
        self.clahe_check.setChecked(False)
        self.clahe_check.setToolTip("CLAHE improves contrast in local regions, useful for uneven illumination")

        # Clip limit slider (controls contrast enhancement strength)
        self.clahe_clip_slider = QSlider(Qt.Horizontal)
        self.clahe_clip_slider.setRange(1, 50)
        self.clahe_clip_slider.setValue(3)  # Default: 0.03
        self.clahe_clip_label = QLabel("0.03")

        # Grid size spinner (controls how local the enhancement is)
        self.clahe_grid_spin = QSpinBox()
        self.clahe_grid_spin.setRange(4, 32)
        self.clahe_grid_spin.setValue(8)  # Default: 8x8 tiles
        self.clahe_grid_spin.setSingleStep(4)
        rb_layout.addWidget(QLabel("Rolling ball radius:"))
        self.rb_slider = QSlider(Qt.Horizontal)
        self.rb_slider.setRange(5, 150)
        self.rb_slider.setValue(50)
        rb_layout.addWidget(self.rb_slider)
        self.rb_spinbox = QSpinBox()
        self.rb_spinbox.setRange(5, 150)
        self.rb_spinbox.setValue(50)
        self.rb_slider.valueChanged.connect(self.rb_spinbox.setValue)
        self.rb_spinbox.valueChanged.connect(self.rb_slider.setValue)
        rb_layout.addWidget(self.rb_spinbox)
        param_layout.addLayout(rb_layout)

        # Additional processing options
        extra_processing_group = QGroupBox("Additional Processing (Optional)")
        extra_processing_group.setMinimumHeight(150)
        extra_layout = QVBoxLayout()

        # Denoising
        self.denoise_check = QCheckBox("Apply Denoising (Median Filter)")
        self.denoise_check.setChecked(False)
        extra_layout.addWidget(self.denoise_check)

        denoise_layout = QHBoxLayout()
        denoise_layout.addWidget(QLabel("  Denoise size:"))
        self.denoise_spin = QSpinBox()
        self.denoise_spin.setRange(3, 7)
        self.denoise_spin.setValue(3)
        self.denoise_spin.setSingleStep(2)
        denoise_layout.addWidget(self.denoise_spin)
        denoise_layout.addWidget(QLabel("(3=gentle, 7=strong)"))
        denoise_layout.addStretch()
        extra_layout.addLayout(denoise_layout)

        # Sharpening
        self.sharpen_check = QCheckBox("Apply Sharpening (Unsharp Mask)")
        self.sharpen_check.setChecked(False)
        extra_layout.addWidget(self.sharpen_check)

        sharpen_layout = QHBoxLayout()
        sharpen_layout.addWidget(QLabel("  Sharpen amount:"))
        self.sharpen_slider = QSlider(Qt.Horizontal)
        self.sharpen_slider.setRange(10, 20)
        self.sharpen_slider.setValue(13)
        sharpen_layout.addWidget(self.sharpen_slider)
        self.sharpen_label = QLabel("1.3")
        self.sharpen_slider.valueChanged.connect(lambda v: self.sharpen_label.setText(f"{v / 10:.1f}"))
        sharpen_layout.addWidget(self.sharpen_label)
        extra_layout.addLayout(sharpen_layout)

        extra_processing_group.setLayout(extra_layout)
        param_layout.addWidget(extra_processing_group)

        self.preview_btn = QPushButton("Preview Current Image")
        self.preview_btn.clicked.connect(self.preview_current_image)
        param_layout.addWidget(self.preview_btn)
        param_group.setLayout(param_layout)
        layout.addWidget(param_group)
        batch_group = QGroupBox("3. Batch Processing")
        batch_layout = QVBoxLayout()
        self.process_selected_btn = QPushButton("Process Selected Images")
        self.process_selected_btn.clicked.connect(self.process_selected_images)
        self.process_selected_btn.setEnabled(False)
        batch_layout.addWidget(self.process_selected_btn)
        self.batch_pick_somas_btn = QPushButton("Pick Somas (All Images)")
        self.batch_pick_somas_btn.clicked.connect(self.start_batch_soma_picking)
        self.batch_pick_somas_btn.setEnabled(False)
        batch_layout.addWidget(self.batch_pick_somas_btn)
        self.batch_outline_btn = QPushButton("Outline Somas (All)")
        self.batch_outline_btn.clicked.connect(self.start_batch_outlining)
        self.batch_outline_btn.setEnabled(False)
        batch_layout.addWidget(self.batch_outline_btn)

        # Add Redo Last Outline button
        self.redo_outline_btn = QPushButton("↩ Redo Last Outline")
        self.redo_outline_btn.clicked.connect(self.redo_last_outline)
        self.redo_outline_btn.setEnabled(False)
        self.redo_outline_btn.setStyleSheet("background-color: #FFE4B5;")
        batch_layout.addWidget(self.redo_outline_btn)

        self.batch_generate_masks_btn = QPushButton("Generate All Masks")
        self.batch_generate_masks_btn.clicked.connect(self.batch_generate_masks)
        self.batch_generate_masks_btn.setEnabled(False)
        batch_layout.addWidget(self.batch_generate_masks_btn)

        # Add Clear All Masks button
        self.clear_masks_btn = QPushButton("🗑 Clear All Masks")
        self.clear_masks_btn.clicked.connect(self.clear_all_masks)
        self.clear_masks_btn.setEnabled(False)
        self.clear_masks_btn.setStyleSheet("background-color: #FFB6C1;")
        batch_layout.addWidget(self.clear_masks_btn)
        self.batch_qa_btn = QPushButton("QA All Masks")
        self.batch_qa_btn.clicked.connect(self.start_batch_qa)
        self.batch_qa_btn.setEnabled(False)
        batch_layout.addWidget(self.batch_qa_btn)
        self.batch_calculate_btn = QPushButton("Calculate Simple Characteristics")
        self.batch_calculate_btn.clicked.connect(self.batch_calculate_morphology)
        self.batch_calculate_btn.setEnabled(False)
        batch_layout.addWidget(self.batch_calculate_btn)
        batch_group.setLayout(batch_layout)
        layout.addWidget(batch_group)
        log_group = QGroupBox("Log")
        log_layout = QVBoxLayout()
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(150)
        log_layout.addWidget(self.log_text)
        log_group.setLayout(log_layout)
        layout.addWidget(log_group)

        # Legend button
        legend_btn = QPushButton("Legend")
        legend_btn.clicked.connect(self.show_legend)
        layout.addWidget(legend_btn)

        layout.addStretch()
        return panel

    def _create_right_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)  # type: QVBoxLayout
        self.tabs = QTabWidget()
        self.original_label = InteractiveImageLabel(self)
        self.original_label.setText("Load images to begin")
        self.tabs.addTab(self.original_label, "Original")
        self.preview_label = InteractiveImageLabel(self)
        self.preview_label.setText("No preview yet")
        self.tabs.addTab(self.preview_label, "Preview")
        self.processed_label = InteractiveImageLabel(self)
        self.processed_label.setText("No processed images yet")
        self.tabs.addTab(self.processed_label, "Processed")
        self.mask_label = InteractiveImageLabel(self)
        self.mask_label.setText("No masks yet")
        self.tabs.addTab(self.mask_label, "Masks")

        # Give tabs most of the space (stretch factor)
        layout.addWidget(self.tabs, stretch=1)

        # Display adjustments button (no stretch)
        display_adjust_btn = QPushButton("Display Adjustments")
        display_adjust_btn.clicked.connect(self.open_display_adjustments)
        layout.addWidget(display_adjust_btn, stretch=0)

        # Progress bar with timer
        progress_container = QHBoxLayout()
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setMinimumHeight(25)
        progress_container.addWidget(self.progress_bar, stretch=3)

        self.timer_label = QLabel("")
        self.timer_label.setVisible(False)
        self.timer_label.setStyleSheet("font-family: monospace; font-size: 12pt; color: #333; padding: 0 10px;")
        self.timer_label.setMinimumWidth(100)
        self.timer_label.setAlignment(Qt.AlignCenter)
        progress_container.addWidget(self.timer_label, stretch=0)

        layout.addLayout(progress_container)

        self.progress_status_label = QLabel("")
        self.progress_status_label.setVisible(False)
        self.progress_status_label.setStyleSheet("color: #666; font-style: italic;")
        layout.addWidget(self.progress_status_label, stretch=0)

        # Timer for tracking processing time
        from PyQt5.QtCore import QTimer
        self.process_timer = QTimer()
        self.process_timer.timeout.connect(self.update_timer_display)
        self.process_start_time = None
        self.timer_running = False

        # Hidden navigation buttons (for compatibility - not displayed but needed by code)
        self.prev_btn = QPushButton("< Previous")
        self.prev_btn.clicked.connect(self.navigate_previous)
        self.prev_btn.setEnabled(False)
        self.prev_btn.setVisible(False)

        self.next_btn = QPushButton("Next >")
        self.next_btn.clicked.connect(self.navigate_next)
        self.next_btn.setEnabled(False)
        self.next_btn.setVisible(False)

        self.done_btn = QPushButton("Done with Current")
        self.done_btn.clicked.connect(self.done_with_current)
        self.done_btn.setEnabled(False)
        self.done_btn.setVisible(False)

        self.nav_status_label = QLabel("")
        self.nav_status_label.setVisible(False)

        self.approve_mask_btn = QPushButton("Approve (A)")
        self.approve_mask_btn.clicked.connect(self.approve_current_mask)
        self.approve_mask_btn.setEnabled(False)
        self.approve_mask_btn.setVisible(False)

        self.reject_mask_btn = QPushButton("Reject (R)")
        self.reject_mask_btn.clicked.connect(self.reject_current_mask)
        self.reject_mask_btn.setEnabled(False)
        self.reject_mask_btn.setVisible(False)

        return panel

    def update_timer_display(self):
        """Update the timer display during processing"""
        if self.process_start_time is not None:
            elapsed = time.time() - self.process_start_time
            minutes = int(elapsed // 60)
            seconds = int(elapsed % 60)
            self.timer_label.setText(f"{minutes:02d}:{seconds:02d}")

    def start_timer(self):
        """Start the processing timer"""
        self.process_start_time = time.time()
        self.timer_running = True
        self.timer_label.setVisible(True)
        self.timer_label.setText("00:00")
        self.process_timer.start(1000)  # Update every second

    def stop_timer(self):
        """Stop the processing timer"""
        self.process_timer.stop()
        self.timer_running = False
        # Keep the final time visible

    def log(self, message):
        self.log_text.append(str(message))

    def show_legend(self):
        """Display a popup dialog with the workflow status legend"""
        dialog = QMessageBox(self)
        dialog.setWindowTitle("Workflow Status Legend")
        dialog.setIcon(QMessageBox.Information)

        legend_text = """
<h3>Workflow Status Colors:</h3>
<p style='line-height: 1.8;'>
<span style='color: #808080;'><b>⚪ Image(s) Loaded</b></span> - Images have been loaded into the application<br>
<span style='color: #009600;'><b>🟢 Image(s) Processed</b></span> - Background removal completed<br>
<span style='color: #0064C8;'><b>🔵 Somas Selected</b></span> - Soma centers have been marked<br>
<span style='color: #C89600;'><b>🟡 Somas Outlined</b></span> - Soma boundaries have been outlined<br>
<span style='color: #FF8C00;'><b>🟠 Masks Generated</b></span> - Cell masks have been generated<br>
<span style='color: #800080;'><b>🟣 Masks QA'ed</b></span> - Quality assurance review completed<br>
<span style='color: #00B400;'><b>✅ Mask Characteristics Processed</b></span> - Morphological analysis complete
</p>
        """

        dialog.setText(legend_text)
        dialog.setTextFormat(Qt.RichText)
        dialog.exec_()

    def open_display_adjustments(self):
        """Open display adjustments dialog"""
        dialog = QDialog(self)
        dialog.setWindowTitle("Display Adjustments (Visual Only)")
        dialog.setMinimumWidth(400)

        layout = QVBoxLayout(dialog)

        info_label = QLabel(
            "<b>Adjust display for better visibility</b><br>"
            "<i>These adjustments do NOT affect image processing or analysis</i>"
        )
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        # Brightness slider
        brightness_group = QGroupBox("Brightness")
        brightness_layout = QVBoxLayout()
        brightness_slider = QSlider(Qt.Horizontal)
        brightness_slider.setRange(-100, 100)
        brightness_slider.setValue(self.brightness_value)
        brightness_slider.setTickPosition(QSlider.TicksBelow)
        brightness_slider.setTickInterval(25)
        brightness_value_label = QLabel(str(self.brightness_value))
        brightness_value_label.setAlignment(Qt.AlignCenter)
        brightness_layout.addWidget(brightness_slider)
        brightness_layout.addWidget(brightness_value_label)
        brightness_group.setLayout(brightness_layout)
        layout.addWidget(brightness_group)

        # Contrast slider
        contrast_group = QGroupBox("Contrast")
        contrast_layout = QVBoxLayout()
        contrast_slider = QSlider(Qt.Horizontal)
        contrast_slider.setRange(-100, 100)
        contrast_slider.setValue(self.contrast_value)
        contrast_slider.setTickPosition(QSlider.TicksBelow)
        contrast_slider.setTickInterval(25)
        contrast_value_label = QLabel(str(self.contrast_value))
        contrast_value_label.setAlignment(Qt.AlignCenter)
        contrast_layout.addWidget(contrast_slider)
        contrast_layout.addWidget(contrast_value_label)
        contrast_group.setLayout(contrast_layout)
        layout.addWidget(contrast_group)

        # Connect sliders to update labels and live preview
        def update_brightness(value):
            brightness_value_label.setText(str(value))
            self.brightness_value = value
            self.update_display()

        def update_contrast(value):
            contrast_value_label.setText(str(value))
            self.contrast_value = value
            self.update_display()

        brightness_slider.valueChanged.connect(update_brightness)
        contrast_slider.valueChanged.connect(update_contrast)

        # Buttons
        button_layout = QHBoxLayout()

        reset_btn = QPushButton("Reset")

        def reset_values():
            brightness_slider.setValue(0)
            contrast_slider.setValue(0)

        reset_btn.clicked.connect(reset_values)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.accept)

        button_layout.addWidget(reset_btn)
        button_layout.addStretch()
        button_layout.addWidget(close_btn)
        layout.addLayout(button_layout)

        dialog.exec_()

    def reset_display_adjustments(self):
        """Reset brightness and contrast to default"""
        self.brightness_value = 0
        self.contrast_value = 0
        self.update_display()

    def update_display(self):
        """Update the display with current brightness/contrast settings"""
        try:
            # Refresh the current image display
            if self.current_image_name and self.current_image_name in self.images:
                current_tab = self.tabs.currentIndex()
                img_data = self.images[self.current_image_name]

                # Determine which image to display based on current tab
                if current_tab == 0:  # Original
                    if 'raw_path' in img_data:
                        raw_img = load_tiff_image(img_data['raw_path'])
                        raw_img = ensure_grayscale(raw_img)
                        adjusted = self._apply_display_adjustments(raw_img)
                        pixmap = self._array_to_pixmap(adjusted, skip_rescale=True)
                        self.original_label.set_image(pixmap)
                elif current_tab == 1:  # Preview
                    if 'preview' in img_data and img_data['preview'] is not None:
                        adjusted = self._apply_display_adjustments(img_data['preview'])
                        pixmap = self._array_to_pixmap(adjusted, skip_rescale=True)
                        self.preview_label.set_image(pixmap)
                elif current_tab == 2:  # Processed
                    if img_data['processed'] is not None:
                        adjusted = self._apply_display_adjustments(img_data['processed'])
                        pixmap = self._array_to_pixmap(adjusted, skip_rescale=True)
                        # Preserve soma markers if in soma picking mode
                        if self.soma_mode:
                            self.processed_label.set_image(pixmap, centroids=img_data['somas'])
                        # Preserve polygon if in outlining mode
                        elif self.processed_label.polygon_mode:
                            queue_idx = len([data for img in self.images.values() for data in img['soma_outlines']])
                            if queue_idx < len(self.outlining_queue):
                                img_name, soma_idx = self.outlining_queue[queue_idx]
                                if img_name == self.current_image_name:
                                    soma = img_data['somas'][soma_idx]
                                    self.processed_label.set_image(pixmap, centroids=[soma],
                                                                   polygon_pts=self.polygon_points)
                        else:
                            self.processed_label.set_image(pixmap, centroids=img_data['somas'])
        except Exception as e:
            self.log(f"ERROR updating display: {str(e)}")
            import traceback
            traceback.print_exc()

    def _apply_display_adjustments(self, img):
        """Apply brightness and contrast adjustments for display only (does not modify original data)"""
        if img is None:
            return None

        # Work with a copy so we don't modify the original
        adjusted = img.astype(np.float32).copy()

        # Normalize to 0-255 range first
        img_min = adjusted.min()
        img_max = adjusted.max()
        if img_max > img_min:
            adjusted = (adjusted - img_min) / (img_max - img_min) * 255.0

        # Apply contrast first (multiply around midpoint)
        contrast = self.contrast_value
        if contrast != 0:
            # Convert contrast from -100,100 to a multiplier
            # -100 = 0.1x (much less contrast), 0 = 1x (no change), 100 = 3x (much more contrast)
            if contrast > 0:
                factor = 1.0 + (contrast / 100.0) * 2.0  # 0 to +2 (max 3x)
            else:
                factor = 1.0 + (contrast / 100.0) * 0.9  # -0.9 to 0 (min 0.1x)

            midpoint = 127.5
            adjusted = (adjusted - midpoint) * factor + midpoint

        # Apply brightness (simple addition)
        brightness = self.brightness_value
        if brightness != 0:
            # Scale brightness to have more visible effect
            adjusted = adjusted + (brightness * 1.5)

        # Clip to valid range
        adjusted = np.clip(adjusted, 0, 255)

        return adjusted.astype(np.uint8)

    def select_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select Image Folder",
            options=QFileDialog.DontUseNativeDialog
        )
        if not folder:
            return
        # Include both lowercase and uppercase extensions for macOS compatibility
        exts = ['*.tif', '*.tiff', '*.png', '*.jpg', '*.jpeg',
                '*.TIF', '*.TIFF', '*.PNG', '*.JPG', '*.JPEG']
        files = []
        for ext in exts:
            files.extend(glob.glob(os.path.join(folder, ext)))
        # Remove duplicates in case filesystem is case-insensitive
        files = list(set(files))
        self.images = {}
        self.file_list.clear()

        from PyQt5.QtGui import QColor, QBrush

        for f in sorted(files):
            img_name = os.path.basename(f)
            self.images[img_name] = {
                'raw_path': f,
                'processed': None,
                'rolling_ball_radius': self.default_rolling_ball_radius,
                'somas': [],
                'soma_ids': [],
                'soma_outlines': [],
                'masks': [],
                'status': 'loaded',
                'selected': False,
                'animal_id': '',
                'treatment': ''
            }
            item = QListWidgetItem(f"☐ ⚪ {img_name} [loaded]")
            item.setData(Qt.UserRole, img_name)
            item.setCheckState(Qt.Unchecked)
            item.setForeground(QBrush(QColor(128, 128, 128)))  # Gray for loaded
            self.file_list.addItem(item)

        if self.images:
            self.process_selected_btn.setEnabled(True)
            self.log(f"Loaded {len(self.images)} images")
            # self.update_workflow_status()

            # Automatically load and display the first image
            first_image_name = sorted(self.images.keys())[0]
            self.current_image_name = first_image_name
            self._display_current_image()

            # Select the first item in the list
            self.file_list.setCurrentRow(0)

            self.log(f"Displaying: {first_image_name}")

    def select_output(self):
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select Output Folder",
            options=QFileDialog.DontUseNativeDialog
        )
        if folder:
            self.output_dir = folder
            self.masks_dir = os.path.join(folder, "masks")
            self.somas_dir = os.path.join(folder, "somas")
            os.makedirs(self.masks_dir, exist_ok=True)
            os.makedirs(self.somas_dir, exist_ok=True)
            self.log(f"Output folder: {folder}")
            self.log(f"Masks will be saved to: {self.masks_dir}")
            self.log(f"Somas will be saved to: {self.somas_dir}")

    def select_all_images(self):
        """Select all images and check all checkboxes"""
        for i in range(self.file_list.count()):
            item = self.file_list.item(i)
            item.setCheckState(Qt.Checked)  # Check the box
            img_name = item.data(Qt.UserRole)
            self.images[img_name]['selected'] = True
        self.log(f"Selected all {self.file_list.count()} images")
        # self.update_workflow_status()

    def clear_all_images(self):
        """Deselect all images and uncheck all checkboxes"""
        for i in range(self.file_list.count()):
            item = self.file_list.item(i)
            item.setCheckState(Qt.Unchecked)  # Uncheck the box
            img_name = item.data(Qt.UserRole)
            self.images[img_name]['selected'] = False
        self.log(f"Cleared selection for all images")
        # self.update_workflow_status()
        # self.update_workflow_status()

    def on_item_checkbox_changed(self, item):
        """Handle checkbox state changes for images in the list"""
        img_name = item.data(Qt.UserRole)
        if img_name and img_name in self.images:
            is_checked = item.checkState() == Qt.Checked
            self.images[img_name]['selected'] = is_checked

    def on_image_selected(self, item):
        img_name = item.data(Qt.UserRole)
        is_checked = item.checkState() == Qt.Checked
        self.images[img_name]['selected'] = is_checked
        self.current_image_name = img_name
        self._display_current_image()
        # self.update_workflow_status()

    def _display_current_image(self):
        if not self.current_image_name or self.current_image_name not in self.images:
            return
        try:
            img_data = self.images[self.current_image_name]
            raw_img = load_tiff_image(img_data['raw_path'])
            raw_img = ensure_grayscale(raw_img)
            # Apply display adjustments
            adjusted_raw = self._apply_display_adjustments(raw_img)
            pixmap = self._array_to_pixmap(adjusted_raw, skip_rescale=True)
            self.original_label.set_image(pixmap)
            if img_data['processed'] is not None:
                # Apply display adjustments to processed image too
                adjusted_proc = self._apply_display_adjustments(img_data['processed'])
                pixmap_proc = self._array_to_pixmap(adjusted_proc, skip_rescale=True)
                self.processed_label.set_image(pixmap_proc, centroids=img_data['somas'])
            else:
                self.processed_label.set_image(self._create_blank_pixmap())
                self.processed_label.setText("Not processed yet")
        except Exception as e:
            self.log(f"ERROR displaying image: {str(e)}")
            import traceback
            traceback.print_exc()

    def _create_blank_pixmap(self):
        blank = np.ones((500, 500), dtype=np.uint8) * 128
        return self._array_to_pixmap(blank)

    def _array_to_pixmap(self, arr, skip_rescale=False):
        arr_disp = arr.astype(float)

        # Only rescale if not already adjusted
        if not skip_rescale:
            arr_disp -= arr_disp.min()
            if arr_disp.max() > 0:
                arr_disp = arr_disp / arr_disp.max() * 255

        arr8 = arr_disp.clip(0, 255).astype(np.uint8)
        arr8 = np.ascontiguousarray(arr8)
        h, w = arr8.shape
        bytes_per_line = w
        img = QImage(arr8.data, w, h, bytes_per_line, QImage.Format_Grayscale8)
        img = img.copy()
        return QPixmap.fromImage(img)

    def preview_current_image(self):
        if not self.current_image_name:
            QMessageBox.warning(self, "Warning", "Select an image first")
            return
        img_data = self.images[self.current_image_name]
        radius = self.rb_slider.value()
        raw_img = load_tiff_image(img_data['raw_path'])
        raw_img = ensure_grayscale(raw_img)
        background = restoration.rolling_ball(raw_img, radius=radius)
        result = raw_img - background
        result = np.clip(result, 0, raw_img.max())
        if self.clahe_check.isChecked():
            clahe_clip = self.clahe_clip_slider.value() / 100.0
            clahe_grid = self.clahe_grid_spin.value()
            img_dtype = result.dtype
            if img_dtype == np.uint16:
                result_clahe = exposure.equalize_adapthist(
                    result,
                    kernel_size=clahe_grid,
                    clip_limit=clahe_clip
                )
                result = (result_clahe * 65535).astype(np.uint16)
            else:
                result_clahe = exposure.equalize_adapthist(
                    result,
                    kernel_size=clahe_grid,
                    clip_limit=clahe_clip
                )
                result = (result_clahe * 255).astype(np.uint8)
        # Apply optional denoising
        if self.denoise_check.isChecked():
            denoise_size = self.denoise_spin.value()
            result = ndimage.median_filter(result, size=denoise_size)

        # Apply optional sharpening
        if self.sharpen_check.isChecked():
            sharpen_amount = self.sharpen_slider.value() / 10.0
            blurred = ndimage.gaussian_filter(result.astype(np.float32), sigma=2)
            result_float = result.astype(np.float32)
            sharpened = result_float + sharpen_amount * (result_float - blurred)
            result = np.clip(sharpened, 0, raw_img.max()).astype(result.dtype)

        # Store the preview (without adjustments)
        img_data['preview'] = result
        # Apply display adjustments for viewing
        adjusted = self._apply_display_adjustments(result)
        pixmap = self._array_to_pixmap(adjusted, skip_rescale=True)
        self.preview_label.set_image(pixmap)
        self.tabs.setCurrentIndex(1)
        steps = [f"RB={radius}"]
        if self.denoise_check.isChecked():
            steps.append(f"Denoise={self.denoise_spin.value()}")
        if self.sharpen_check.isChecked():
            steps.append(f"Sharpen={self.sharpen_slider.value() / 10:.1f}")
        self.log(f"Preview {self.current_image_name}: {', '.join(steps)}")

    def process_selected_images(self):
        if not self.output_dir:
            QMessageBox.warning(self, "Warning", "Select output folder first")
            return
        selected_images = [(name, data) for name, data in self.images.items() if data['selected']]
        if not selected_images:
            QMessageBox.warning(self, "Warning", "No images selected")
            return

        radius = self.rb_slider.value()
        denoise_enabled = self.denoise_check.isChecked()
        denoise_size = self.denoise_spin.value()
        sharpen_enabled = self.sharpen_check.isChecked()
        sharpen_amount = self.sharpen_slider.value() / 10.0

        process_list = []
        for img_name, img_data in selected_images:
            process_list.append((img_data['raw_path'], img_name, radius,
                                 denoise_enabled, denoise_size, sharpen_enabled, sharpen_amount))
        clahe_enabled = self.clahe_check.isChecked()
        clahe_clip = self.clahe_clip_slider.value()
        clahe_grid = self.clahe_grid_spin.value()

        process_list.append((img_data['raw_path'], img_name, radius,
                             denoise_enabled, denoise_size, sharpen_enabled, sharpen_amount,
                             clahe_enabled, clahe_clip, clahe_grid))  # Added CLAHE params
        if self.clahe_check.isChecked():
            steps.append(f"CLAHE(clip={self.clahe_clip_slider.value() / 100:.2f}, grid={self.clahe_grid_spin.value()})")
        self.thread = BackgroundRemovalThread(process_list, self.output_dir)
        self.thread.status_update.connect(self.log)
        self.thread.progress.connect(self._update_progress)
        self.thread.finished_image.connect(self._handle_processed_image)
        self.thread.finished.connect(self._background_removal_finished)
        self.thread.error_occurred.connect(lambda msg: self.log(f"ERROR: {msg}"))
        self.progress_bar.setVisible(True)
        self.progress_status_label.setVisible(True)
        self.progress_status_label.setText("Processing images...")
        self.process_selected_btn.setEnabled(False)
        self.thread.start()

    def _update_progress(self, value):
        self.progress_bar.setValue(value)

    def _handle_processed_image(self, output_path, img_name, processed_data):
        if img_name in self.images:
            self.images[img_name]['processed'] = processed_data
            self.images[img_name]['status'] = 'processed'
            self._update_file_list_item(img_name)
            if img_name == self.current_image_name:
                pixmap = self._array_to_pixmap(processed_data)
                self.processed_label.set_image(pixmap)
                self.tabs.setCurrentIndex(2)

    def _update_file_list_item(self, img_name):
        for i in range(self.file_list.count()):
            item = self.file_list.item(i)
            if item.data(Qt.UserRole) == img_name:
                check_mark = "☑" if self.images[img_name]['selected'] else "☐"
                status = self.images[img_name]['status']

                # Add visual indicators for different statuses
                status_icons = {
                    'loaded': '⚪',
                    'processed': '🟢',
                    'somas_picked': '🔵',
                    'outlined': '🟡',
                    'masks_generated': '🟠',
                    'qa_complete': '🟣',
                    'analyzed': '✅'
                }

                status_icon = status_icons.get(status, '⚪')
                item.setText(f"{check_mark} {status_icon} {img_name} [{status}]")

                # Color code the item based on status
                from PyQt5.QtGui import QColor, QBrush
                if status == 'loaded':
                    item.setForeground(QBrush(QColor(128, 128, 128)))  # Gray
                elif status == 'processed':
                    item.setForeground(QBrush(QColor(0, 150, 0)))  # Green
                elif status == 'somas_picked':
                    item.setForeground(QBrush(QColor(0, 100, 200)))  # Blue
                elif status == 'outlined':
                    item.setForeground(QBrush(QColor(200, 150, 0)))  # Yellow/Gold
                elif status == 'masks_generated':
                    item.setForeground(QBrush(QColor(255, 140, 0)))  # Orange
                elif status == 'qa_complete':
                    item.setForeground(QBrush(QColor(128, 0, 128)))  # Purple
                elif status == 'analyzed':
                    item.setForeground(QBrush(QColor(0, 180, 0)))  # Bright Green
                    item.setBackground(QBrush(QColor(230, 255, 230)))  # Light green background
                break

    def _background_removal_finished(self):

        self.progress_bar.setVisible(False)
        self.progress_status_label.setVisible(False)
        self.process_selected_btn.setEnabled(True)
        self.batch_pick_somas_btn.setEnabled(True)
        self.log("=" * 50)
        self.log("✓ Background removal complete!")
        self.log("✓ Ready for batch soma picking")
        self.log("=" * 50)
        # self.update_workflow_status()
        QMessageBox.information(
            self, "Complete",
            "All selected images processed!\n\nYou can now pick somas."
        )

    def update_workflow_status(self):
        selected = [name for name, data in self.images.items() if data['selected']]
        if not selected:
            self.workflow_status_label.setText("No images selected")
            return
        status_counts = {}
        for img_name in selected:
            status = self.images[img_name]['status']
            status_counts[status] = status_counts.get(status, 0) + 1
        status_text = f"Selected: {len(selected)} images\n"
        for status, count in sorted(status_counts.items()):
            status_text += f"{status}: {count}\n"
        self.workflow_status_label.setText(status_text)

    def get_current_processed_image(self):
        if not self.current_image_name or self.current_image_name not in self.images:
            return None
        return self.images[self.current_image_name]['processed']

    def start_batch_soma_picking(self):
        self.soma_picking_queue = [name for name, data in self.images.items()
                                   if data['selected'] and data['status'] == 'processed']
        if not self.soma_picking_queue:
            QMessageBox.warning(self, "Warning", "No processed images to pick somas from")
            return
        self.batch_mode = True
        self.current_image_name = self.soma_picking_queue[0]
        self.processed_label.soma_mode = True
        self.original_label.soma_mode = False
        self.preview_label.soma_mode = False
        self.mask_label.soma_mode = False
        self._load_image_for_soma_picking()
        self.prev_btn.setEnabled(True)
        self.next_btn.setEnabled(True)
        self.done_btn.setEnabled(True)
        self.log("=" * 50)
        self.log("🎯 BATCH SOMA PICKING MODE")
        self.log(f"Click somas on: {self.current_image_name}")
        self.log("Click 'Done with Current' when finished with this image")
        self.log("=" * 50)

    def _load_image_for_soma_picking(self):
        if not self.current_image_name:
            return
        img_data = self.images[self.current_image_name]
        pixmap = self._array_to_pixmap(img_data['processed'])
        self.processed_label.set_image(pixmap, centroids=img_data['somas'])
        self.tabs.setCurrentIndex(2)
        current_idx = self.soma_picking_queue.index(
            self.current_image_name) if self.current_image_name in self.soma_picking_queue else -1
        self.nav_status_label.setText(
            f"Image {current_idx + 1}/{len(self.soma_picking_queue)}: {self.current_image_name} | "
            f"Somas: {len(img_data['somas'])}"
        )

    def add_soma(self, coords):
        if not self.current_image_name:
            return
        img_data = self.images[self.current_image_name]
        img_data['somas'].append(coords)
        soma_id = f"soma_{coords[0]}_{coords[1]}"
        img_data['soma_ids'].append(soma_id)
        pixmap = self._array_to_pixmap(img_data['processed'])
        self.processed_label.set_image(pixmap, centroids=img_data['somas'])
        self.log(f"✓ {self.current_image_name}: Soma {len(img_data['somas'])} added | ID: {soma_id}")
        self._load_image_for_soma_picking()

    def done_with_current(self):
        if not self.current_image_name:
            return
        img_data = self.images[self.current_image_name]
        if len(img_data['somas']) > 0:
            img_data['status'] = 'somas_picked'
            self._update_file_list_item(self.current_image_name)
        self.navigate_next()

    def navigate_next(self):
        if not self.soma_picking_queue:
            return
        current_idx = self.soma_picking_queue.index(
            self.current_image_name) if self.current_image_name in self.soma_picking_queue else -1
        if current_idx < len(self.soma_picking_queue) - 1:
            self.current_image_name = self.soma_picking_queue[current_idx + 1]
            self._load_image_for_soma_picking()
        else:
            self._finish_soma_picking()

    def navigate_previous(self):
        if not self.soma_picking_queue:
            return
        current_idx = self.soma_picking_queue.index(
            self.current_image_name) if self.current_image_name in self.soma_picking_queue else 0
        if current_idx > 0:
            self.current_image_name = self.soma_picking_queue[current_idx - 1]
            self._load_image_for_soma_picking()

    def _finish_soma_picking(self):
        self.batch_mode = False
        self.processed_label.soma_mode = False
        self.prev_btn.setEnabled(False)
        self.next_btn.setEnabled(False)
        self.done_btn.setEnabled(False)
        total_somas = sum(len(data['somas']) for data in self.images.values() if data['selected'])
        self.batch_outline_btn.setEnabled(True)
        # self.update_workflow_status()
        self.log("=" * 50)
        self.log(f"✓ Soma picking complete! Total somas: {total_somas}")
        self.log("✓ Ready for outlining")
        self.log("=" * 50)
        QMessageBox.information(
            self, "Complete",
            f"Soma picking complete!\n\nTotal somas marked: {total_somas}\n\nReady to outline."
        )

    def start_batch_outlining(self):
        self.outlining_queue = []
        for img_name, img_data in self.images.items():
            if not img_data['selected'] or img_data['status'] != 'somas_picked':
                continue
            for soma_idx in range(len(img_data['somas'])):
                self.outlining_queue.append((img_name, soma_idx))
        if not self.outlining_queue:
            QMessageBox.warning(self, "Warning", "No somas to outline")
            return
        self.batch_mode = True
        self.polygon_points = []
        self.processed_label.polygon_mode = True
        self.processed_label.soma_mode = False
        self.original_label.polygon_mode = False
        self.preview_label.polygon_mode = False
        self.mask_label.polygon_mode = False
        self._load_soma_for_outlining(0)
        self.prev_btn.setEnabled(False)
        self.next_btn.setEnabled(False)
        self.done_btn.setEnabled(False)
        self.log("=" * 50)
        self.log(f"📐 BATCH OUTLINING MODE")
        self.log(f"Total somas to outline: {len(self.outlining_queue)}")
        self.log("✓ Auto-zoom enabled - each soma will be zoomed to 3x")
        self.log("Mouse Controls:")
        self.log("  • Left-click: add point | Right-click or Double-click: finish outline")
        self.log("  • Mouse wheel: zoom in/out")
        self.log("Keyboard Shortcuts:")
        self.log("  • Z/Backspace: undo last point | Escape: restart | Enter: finish")
        self.log("  • Home/0: reset zoom to 1x")
        self.log("=" * 50)

    def _load_soma_for_outlining(self, queue_idx):
        if queue_idx >= len(self.outlining_queue):
            self._finish_outlining()
            return
        img_name, soma_idx = self.outlining_queue[queue_idx]
        self.current_image_name = img_name
        img_data = self.images[img_name]
        soma = img_data['somas'][soma_idx]
        soma_id = img_data['soma_ids'][soma_idx]
        pixmap = self._array_to_pixmap(img_data['processed'])
        self.processed_label.set_image(pixmap, centroids=[soma], polygon_pts=self.polygon_points)

        # Auto-zoom to the soma for easier outlining
        self.processed_label.auto_zoom_to_point(soma, zoom_level=3.0)

        self.tabs.setCurrentIndex(2)
        self.nav_status_label.setText(
            f"Soma {queue_idx + 1}/{len(self.outlining_queue)} | "
            f"Image: {img_name} | ID: {soma_id}"
        )
        self.log(f"Outlining {soma_id} ({queue_idx + 1}/{len(self.outlining_queue)})")

    def add_polygon_point(self, coords):
        self.polygon_points.append(coords)
        queue_idx = len([data for img_data in self.images.values() for data in img_data['soma_outlines']])
        if queue_idx < len(self.outlining_queue):
            img_name, soma_idx = self.outlining_queue[queue_idx]
            img_data = self.images[img_name]
            soma = img_data['somas'][soma_idx]
            soma_id = img_data['soma_ids'][soma_idx]
            pixmap = self._array_to_pixmap(img_data['processed'])
            self.processed_label.set_image(pixmap, centroids=[soma], polygon_pts=self.polygon_points)
            # Update status to show point count
            self.nav_status_label.setText(
                f"Soma {queue_idx + 1}/{len(self.outlining_queue)} | "
                f"Image: {img_name} | ID: {soma_id} | Points: {len(self.polygon_points)}"
            )

    def undo_last_polygon_point(self):
        """Remove the last point added to the polygon"""
        if len(self.polygon_points) > 0:
            self.polygon_points.pop()
            self.log(f"↩️ Undid last point ({len(self.polygon_points)} points remaining)")
            # Refresh the display
            queue_idx = len([data for img_data in self.images.values() for data in img_data['soma_outlines']])
            if queue_idx < len(self.outlining_queue):
                img_name, soma_idx = self.outlining_queue[queue_idx]
                img_data = self.images[img_name]
                soma = img_data['somas'][soma_idx]
                soma_id = img_data['soma_ids'][soma_idx]
                pixmap = self._array_to_pixmap(img_data['processed'])
                self.processed_label.set_image(pixmap, centroids=[soma], polygon_pts=self.polygon_points)
                self.nav_status_label.setText(
                    f"Soma {queue_idx + 1}/{len(self.outlining_queue)} | "
                    f"Image: {img_name} | ID: {soma_id} | Points: {len(self.polygon_points)}"
                )
        else:
            self.log("⚠️ No points to undo")

    def restart_polygon(self):
        """Clear all points and restart the current outline"""
        if len(self.polygon_points) > 0:
            self.polygon_points = []
            self.log("🔄 Restarted outline (all points cleared)")
            # Refresh the display
            queue_idx = len([data for img_data in self.images.values() for data in img_data['soma_outlines']])
            if queue_idx < len(self.outlining_queue):
                img_name, soma_idx = self.outlining_queue[queue_idx]
                img_data = self.images[img_name]
                soma = img_data['somas'][soma_idx]
                soma_id = img_data['soma_ids'][soma_idx]
                pixmap = self._array_to_pixmap(img_data['processed'])
                self.processed_label.set_image(pixmap, centroids=[soma], polygon_pts=self.polygon_points)
                self.nav_status_label.setText(
                    f"Soma {queue_idx + 1}/{len(self.outlining_queue)} | "
                    f"Image: {img_name} | ID: {soma_id} | Points: {len(self.polygon_points)}"
                )
        else:
            self.log("⚠️ No points to clear")

    def finish_polygon(self):
        if len(self.polygon_points) < 3:
            QMessageBox.warning(self, "Warning", "Need at least 3 points")
            return
        queue_idx = len([data for img_data in self.images.values() for data in img_data['soma_outlines']])
        if queue_idx >= len(self.outlining_queue):
            return
        img_name, soma_idx = self.outlining_queue[queue_idx]
        img_data = self.images[img_name]
        mask = self._polygon_to_mask(self.polygon_points, img_data['processed'].shape)
        soma_id = img_data['soma_ids'][soma_idx]

        # Calculate soma area from the outline
        pixel_size = float(self.pixel_size_input.text())
        soma_area_um2 = np.sum(mask) * (pixel_size ** 2)

        img_data['soma_outlines'].append({
            'soma_idx': soma_idx,
            'soma_id': soma_id,
            'centroid': img_data['somas'][soma_idx],
            'outline': mask,
            'polygon_points': self.polygon_points.copy(),
            'soma_area_um2': soma_area_um2  # Store soma area
        })

        # Export soma outline to file
        self._export_soma_outline(img_name, soma_id, mask, pixel_size, soma_area_um2)

        self.log(f"✓ {soma_id} outlined (soma area: {soma_area_um2:.1f} µm²)")
        self.polygon_points = []

        # Enable Redo button after first outline is complete
        self.redo_outline_btn.setEnabled(True)

        self._load_soma_for_outlining(queue_idx + 1)

    def redo_last_outline(self):
        """Delete the last completed outline and go back to redo it"""
        # Find the last outline across all images
        last_outline_img = None
        last_outline_data = None

        for img_name, img_data in self.images.items():
            if img_data['soma_outlines']:
                last_outline_img = img_name
                last_outline_data = img_data['soma_outlines'][-1]

        if not last_outline_data:
            QMessageBox.warning(self, "Warning", "No outlines to redo")
            return

        reply = QMessageBox.question(
            self, 'Redo Last Outline',
            f"Delete outline for {last_outline_data['soma_id']} and redo it?",
            QMessageBox.Yes | QMessageBox.No
        )

        if reply == QMessageBox.No:
            return

        # Remove the last outline
        self.images[last_outline_img]['soma_outlines'].pop()

        # Delete the exported soma file if it exists
        if self.masks_dir:
            soma_id = last_outline_data['soma_id']
            soma_file = os.path.join(self.masks_dir, f"{os.path.splitext(last_outline_img)[0]}_{soma_id}_soma.tif")
            if os.path.exists(soma_file):
                os.remove(soma_file)

        self.log(f"↩ Deleted outline for {last_outline_data['soma_id']} - ready to redo")

        # Go back to that soma for re-outlining
        queue_idx = len([data for img_data in self.images.values() for data in img_data['soma_outlines']])
        self.polygon_points = []

        # If no more outlines left, disable the redo button
        if queue_idx == 0:
            self.redo_outline_btn.setEnabled(False)

        self._load_soma_for_outlining(queue_idx)

    def clear_all_masks(self):
        """Delete all generated masks and allow regeneration"""
        # Count total masks
        total_masks = sum(len(data['masks']) for data in self.images.values())

        if total_masks == 0:
            QMessageBox.information(self, "Info", "No masks to clear")
            return

        reply = QMessageBox.question(
            self, 'Clear All Masks',
            f"Delete all {total_masks} generated masks?\n\n"
            f"This will allow you to regenerate masks with different settings.",
            QMessageBox.Yes | QMessageBox.No
        )

        if reply == QMessageBox.No:
            return

        # Clear masks from all images
        masks_cleared = 0
        for img_name, img_data in self.images.items():
            if img_data['masks']:
                masks_cleared += len(img_data['masks'])
                img_data['masks'] = []
                # Update status back to 'outlined'
                if img_data['status'] in ['masks_generated', 'qa_complete', 'morphology_calculated']:
                    img_data['status'] = 'outlined'
                    self._update_file_list_item(img_name)

        # Delete mask files from disk if they exist
        if self.masks_dir and os.path.exists(self.masks_dir):
            mask_files = glob.glob(os.path.join(self.masks_dir, "*_mask.tif"))
            for mask_file in mask_files:
                try:
                    os.remove(mask_file)
                except Exception as e:
                    self.log(f"Warning: Could not delete {os.path.basename(mask_file)}: {e}")

        # Disable buttons that depend on masks
        self.batch_qa_btn.setEnabled(False)
        self.batch_calculate_btn.setEnabled(False)
        self.clear_masks_btn.setEnabled(False)

        # Re-enable mask generation
        self.batch_generate_masks_btn.setEnabled(True)

        self.log("=" * 50)
        self.log(f"🗑 Cleared {masks_cleared} masks")
        self.log("✓ Ready to regenerate masks with new settings")
        self.log("=" * 50)

    def _polygon_to_mask(self, polygon, shape):
        if len(polygon) < 3:
            return np.zeros(shape, dtype=np.uint8)
        poly_array = np.array([[p[1], p[0]] for p in polygon])
        h, w = shape[:2]
        yy, xx = np.mgrid[:h, :w]
        points = np.c_[xx.ravel(), yy.ravel()]
        path = mplPath(poly_array)
        mask = path.contains_points(points).reshape(h, w)
        return mask.astype(np.uint8)

    def _finish_outlining(self):
        self.batch_mode = False
        self.processed_label.polygon_mode = False
        self.redo_outline_btn.setEnabled(False)  # Disable redo button when outlining complete
        for img_name, img_data in self.images.items():
            if img_data['selected'] and len(img_data['soma_outlines']) == len(img_data['somas']):
                img_data['status'] = 'outlined'
                self._update_file_list_item(img_name)
        self.batch_generate_masks_btn.setEnabled(True)
        # self.update_workflow_status()
        self.log("=" * 50)
        self.log("✓ All somas outlined!")
        self.log("✓ Ready to generate masks")
        self.log("=" * 50)
        QMessageBox.information(self, "Complete", "All somas outlined!\n\nReady to generate masks.")

    def batch_generate_masks(self):
        # Show mask generation settings dialog first
        dialog = QDialog(self)
        dialog.setWindowTitle("Mask Generation Settings")
        dialog.setModal(True)

        layout = QVBoxLayout()

        # Title
        title = QLabel("Configure Mask Generation Settings")
        title_font = title.font()
        title_font.setBold(True)
        title_font.setPointSize(12)
        title.setFont(title_font)
        layout.addWidget(title)

        # Description
        desc = QLabel("Adjust settings to prevent dim background from being included in masks:")
        desc.setWordWrap(True)
        layout.addWidget(desc)

        layout.addSpacing(10)

        # Minimum intensity threshold checkbox
        min_intensity_check = QCheckBox("Use minimum intensity threshold")
        min_intensity_check.setChecked(self.use_min_intensity)
        min_intensity_check.setToolTip("Exclude pixels below this intensity from masks")
        layout.addWidget(min_intensity_check)

        # Minimum intensity slider
        slider_layout = QHBoxLayout()
        slider_layout.addWidget(QLabel("  Min intensity:"))
        min_intensity_slider = QSlider(Qt.Horizontal)
        min_intensity_slider.setRange(0, 100)
        min_intensity_slider.setValue(self.min_intensity_percent)
        slider_layout.addWidget(min_intensity_slider)
        min_intensity_label = QLabel(f"{self.min_intensity_percent}%")
        min_intensity_slider.valueChanged.connect(
            lambda v: min_intensity_label.setText(f"{v}%")
        )
        slider_layout.addWidget(min_intensity_label)
        layout.addLayout(slider_layout)

        layout.addSpacing(10)

        # Help text
        help_text = QLabel(
            "💡 Tip:\n"
            "• 0% = No minimum (include all pixels)\n"
            "• 30% = Default (good for most images)\n"
            "• 50% = Strict (exclude dimmer pixels)\n"
            "• 70%+ = Very strict (only bright pixels)"
        )
        help_text.setStyleSheet("background-color: #f0f0f0; padding: 10px; border-radius: 5px;")
        help_text.setWordWrap(True)
        layout.addWidget(help_text)

        layout.addSpacing(10)

        # Buttons
        button_layout = QHBoxLayout()
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(dialog.reject)
        button_layout.addWidget(cancel_btn)

        ok_btn = QPushButton("Generate Masks")
        ok_btn.setDefault(True)
        ok_btn.clicked.connect(dialog.accept)
        ok_btn.setStyleSheet(
            "QPushButton { background-color: #4CAF50; color: white; font-weight: bold; padding: 5px; }")
        button_layout.addWidget(ok_btn)

        layout.addLayout(button_layout)

        dialog.setLayout(layout)
        dialog.setMinimumWidth(400)

        # Show dialog and get result
        if dialog.exec_() != QDialog.Accepted:
            return  # User cancelled

        # Save settings
        self.use_min_intensity = min_intensity_check.isChecked()
        self.min_intensity_percent = min_intensity_slider.value()

        # Now proceed with mask generation
        try:
            pixel_size = float(self.pixel_size_input.text())
            area_list = [200, 300, 400, 500, 600, 700, 800]

            self.progress_bar.setVisible(True)
            self.progress_status_label.setVisible(True)

            total_outlines = sum(len(data['soma_outlines']) for data in self.images.values() if data['selected'])
            current_count = 0

            for img_name, img_data in self.images.items():
                if not img_data['selected'] or img_data['status'] != 'outlined':
                    continue

                self.log(f"Generating masks for {img_name}...")

                for soma_data in img_data['soma_outlines']:
                    centroid = soma_data['centroid']
                    soma_idx = soma_data['soma_idx']
                    soma_id = soma_data['soma_id']
                    soma_area_um2 = soma_data.get('soma_area_um2', 0)  # Get soma area from outline

                    self.progress_status_label.setText(f"Generating masks: {current_count + 1}/{total_outlines}")
                    QApplication.processEvents()

                    masks = self._create_annulus_masks(
                        centroid, area_list, pixel_size, soma_idx, soma_id,
                        img_data['processed'], img_name, soma_area_um2  # Pass soma area
                    )
                    img_data['masks'].extend(masks)

                    current_count += 1
                    self.progress_bar.setValue(int((current_count / total_outlines) * 100))

                img_data['status'] = 'masks_generated'
                self._update_file_list_item(img_name)

            self.progress_bar.setVisible(False)
            self.progress_status_label.setVisible(False)

            self.batch_qa_btn.setEnabled(True)
            self.clear_masks_btn.setEnabled(True)  # Enable clear masks button
            # self.update_workflow_status()

            total_masks = sum(len(data['masks']) for data in self.images.values() if data['selected'])

            self.log("=" * 50)
            self.log(f"✓ Generated {total_masks} masks total")
            if self.use_min_intensity:
                self.log(f"✓ Used minimum intensity: {self.min_intensity_percent}%")
            self.log("✓ Ready for QA")
            self.log("=" * 50)

            QMessageBox.information(
                self, "Success",
                f"Generated {total_masks} masks!\n\nReady for QA."
            )

        except Exception as e:
            self.progress_bar.setVisible(False)
            self.progress_status_label.setVisible(False)
            self.log(f"ERROR: {e}")
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Error", f"Failed: {e}")

    def _create_annulus_masks(self, centroid, area_list_um2, pixel_size_um, soma_idx, soma_id, processed_img, img_name,
                              soma_area_um2):
        """Create cell masks with unique soma ID and constant soma area"""
        from skimage.filters import threshold_otsu
        from skimage.measure import label
        from scipy.ndimage import shift

        masks = []
        cy, cx = int(centroid[0]), int(centroid[1])

        roi_size = 200
        y_min = max(0, cy - roi_size)
        y_max = min(processed_img.shape[0], cy + roi_size)
        x_min = max(0, cx - roi_size)
        x_max = min(processed_img.shape[1], cx + roi_size)

        roi = processed_img[y_min:y_max, x_min:x_max].copy()
        centroid_in_roi = (cy - y_min, cx - x_min)

        reversed_area_list = sorted(area_list_um2, reverse=True)

        for i, target_area_um2 in enumerate(reversed_area_list):
            if target_area_um2 == 200:
                continue

            target_area_px = target_area_um2 / (pixel_size_um ** 2)

            # Calculate minimum intensity threshold if enabled
            # Scale it based on mask size - larger masks get lower minimum
            min_intensity = None
            if self.use_min_intensity:
                # Convert percentage to actual intensity value
                roi_max = roi.max()
                base_min_intensity = (self.min_intensity_percent / 100.0) * roi_max

                # Scale factor: smaller masks use full minimum, larger masks use less
                # 300 µm² (smallest) = 100% of minimum
                # 800 µm² (largest) = 30% of minimum
                smallest_area = 300
                largest_area = 800
                if target_area_um2 <= smallest_area:
                    scale_factor = 1.0
                elif target_area_um2 >= largest_area:
                    scale_factor = 0.3
                else:
                    # Linear interpolation between 1.0 and 0.3
                    scale_factor = 1.0 - (0.7 * (target_area_um2 - smallest_area) / (largest_area - smallest_area))

                min_intensity = base_min_intensity * scale_factor

            mask = self._iterative_threshold_mask(
                roi, centroid_in_roi, target_area_px, max_iterations=30, min_intensity=min_intensity
            )

            if np.any(mask):
                labeled_mask = label(mask)

                if labeled_mask.max() > 0:
                    cy_roi, cx_roi = int(centroid_in_roi[0]), int(centroid_in_roi[1])

                    if (0 <= cy_roi < labeled_mask.shape[0] and
                            0 <= cx_roi < labeled_mask.shape[1] and
                            labeled_mask[cy_roi, cx_roi] > 0):
                        target_label = labeled_mask[cy_roi, cx_roi]
                        mask = (labeled_mask == target_label).astype(np.uint8)
                    else:
                        best_region = None
                        min_dist = float('inf')

                        for region_id in range(1, labeled_mask.max() + 1):
                            region_mask = (labeled_mask == region_id)
                            region_coords = np.argwhere(region_mask)
                            distances = np.sqrt(np.sum((region_coords - np.array(centroid_in_roi)) ** 2, axis=1))
                            closest_dist = np.min(distances)

                            if closest_dist < min_dist:
                                min_dist = closest_dist
                                best_region = region_mask

                        if best_region is not None:
                            mask = best_region.astype(np.uint8)

                    # Don't shift the mask - let it use its natural center
                    # The clicked centroid is just used to select which region, not to force positioning

            full_mask = np.zeros(processed_img.shape, dtype=np.uint8)
            full_mask[y_min:y_max, x_min:x_max] = mask

            # Mask saving now happens during QA approval (see _export_approved_mask)
            # This avoids saving masks that will be rejected
            # if self.masks_dir and np.any(full_mask):
            #     image_name_base = os.path.splitext(img_name)[0]
            #     mask_filename = f"{image_name_base}_{soma_id}_area{int(target_area_um2)}.tif"
            #     mask_path = os.path.join(self.masks_dir, mask_filename)
            #     tifffile.imwrite(mask_path, full_mask)

            masks.append({
                'image_name': img_name,
                'soma_idx': soma_idx,
                'soma_id': soma_id,
                'area_um2': target_area_um2,
                'mask': full_mask,
                'approved': None,
                'soma_area_um2': soma_area_um2  # Store constant soma area from outline
            })

        return masks

    def _iterative_threshold_mask(self, roi, centroid, target_area, max_iterations=30, tolerance=100,
                                  min_intensity=None):
        """Use iterative thresholding to create a mask of target area"""
        from skimage.filters import threshold_otsu
        from skimage.measure import label

        try:
            current_thresh = threshold_otsu(roi)
        except:
            current_thresh = np.percentile(roi, 50)

        # If min_intensity is set, make sure we start at least that high
        if min_intensity is not None:
            current_thresh = max(current_thresh, min_intensity)

        cy, cx = int(centroid[0]), int(centroid[1])

        best_mask = None
        best_diff = float('inf')

        for iteration in range(max_iterations):
            binary = roi > current_thresh
            labeled = label(binary)

            mask = None
            if 0 <= cy < labeled.shape[0] and 0 <= cx < labeled.shape[1]:
                centroid_label = labeled[cy, cx]
                if centroid_label > 0:
                    mask = (labeled == centroid_label).astype(np.uint8)
                else:
                    if labeled.max() > 0:
                        distances = []
                        for region_label in range(1, labeled.max() + 1):
                            region_mask = (labeled == region_label)
                            region_coords = np.argwhere(region_mask)
                            if len(region_coords) > 0:
                                dists = np.sqrt(np.sum((region_coords - np.array([cy, cx])) ** 2, axis=1))
                                min_dist = np.min(dists)
                                distances.append((min_dist, region_label))

                        if distances:
                            _, closest_label = min(distances, key=lambda x: x[0])
                            mask = (labeled == closest_label).astype(np.uint8)

                    if mask is None:
                        mask = binary.astype(np.uint8)
            else:
                mask = binary.astype(np.uint8)

            current_area = np.sum(mask)
            area_diff = abs(current_area - target_area)

            if area_diff < best_diff:
                best_diff = area_diff
                best_mask = mask.copy()

            if area_diff <= tolerance:
                return mask

            adjustment = current_thresh * (current_area - target_area) / (iteration + 1) / target_area
            current_thresh += adjustment

            # Clip threshold, but respect minimum intensity if set
            if min_intensity is not None:
                current_thresh = np.clip(current_thresh, min_intensity, roi.max())
            else:
                current_thresh = np.clip(current_thresh, roi.min(), roi.max())

            if abs(adjustment) < 0.1:
                break

        return best_mask if best_mask is not None else np.zeros_like(roi, dtype=np.uint8)

    def start_batch_qa(self):
        # Flatten all masks from all images
        self.all_masks_flat = []
        for img_name, img_data in self.images.items():
            if not img_data['selected']:
                continue
            for mask_data in img_data['masks']:
                self.all_masks_flat.append({
                    'image_name': img_name,
                    'mask_data': mask_data,
                    'processed_img': img_data['processed']
                })

        if not self.all_masks_flat:
            QMessageBox.warning(self, "Warning", "No masks to QA")
            return

        self.mask_qa_active = True
        self.mask_qa_idx = 0

        self.approve_mask_btn.setEnabled(True)
        self.reject_mask_btn.setEnabled(True)
        self.prev_btn.setEnabled(True)
        self.next_btn.setEnabled(True)
        self.done_btn.setEnabled(False)

        self._show_current_mask()
        self.tabs.setCurrentIndex(3)

        self.log("=" * 50)
        self.log("🎯 BATCH MASK QA MODE")
        self.log(f"Total masks to review: {len(self.all_masks_flat)}")
        self.log("Keyboard: A=Approve, R=Reject, ←→=Navigate, Space=Approve&Next")
        self.log("=" * 50)

    def _show_current_mask(self):
        if not self.all_masks_flat or self.mask_qa_idx >= len(self.all_masks_flat):
            return

        flat_data = self.all_masks_flat[self.mask_qa_idx]
        mask_data = flat_data['mask_data']
        processed_img = flat_data['processed_img']
        img_name = flat_data['image_name']

        pixmap = self._array_to_pixmap(processed_img)
        self.mask_label.set_image(pixmap, mask_overlay=mask_data['mask'])

        status = mask_data.get('approved')
        status_text = "✓ Approved" if status is True else "✗ Rejected" if status is False else "⏳ Not reviewed"

        self.nav_status_label.setText(
            f"Mask {self.mask_qa_idx + 1}/{len(self.all_masks_flat)} | "
            f"{img_name} | {mask_data['soma_id']} | "
            f"Area: {mask_data['area_um2']} µm² | {status_text}"
        )

    def approve_current_mask(self):
        if not self.mask_qa_active or self.mask_qa_idx >= len(self.all_masks_flat):
            return

        flat_data = self.all_masks_flat[self.mask_qa_idx]
        mask_data = flat_data['mask_data']
        mask_data['approved'] = True

        current_soma_id = mask_data['soma_id']
        current_area = mask_data['area_um2']
        current_img = flat_data['image_name']

        self.log(f"✅ APPROVED | {current_img} | {current_soma_id} | Area: {current_area} µm²")

        # Export mask immediately upon approval
        self._export_approved_mask(flat_data)

        # Auto-approve ALL smaller masks from the SAME soma in the SAME image
        auto_approved = []
        for i, other_flat in enumerate(self.all_masks_flat):
            other_mask = other_flat['mask_data']
            other_img = other_flat['image_name']

            # Must be: same image, same soma, smaller area, not yet reviewed
            if (other_img == current_img and
                    other_mask['soma_id'] == current_soma_id and
                    other_mask['area_um2'] < current_area and
                    other_mask['approved'] is None):
                other_mask['approved'] = True
                auto_approved.append((i + 1, other_mask['area_um2']))
                # Export auto-approved masks too
                self._export_approved_mask(other_flat)

        if auto_approved:
            self.log(f"   ⚡ Auto-approved {len(auto_approved)} smaller masks for {current_soma_id}:")
            for mask_num, area in auto_approved:
                self.log(f"      Mask #{mask_num} ({area} µm²)")

        # Move to next unreviewed mask
        self._advance_to_next_unreviewed()

    def _export_approved_mask(self, flat_data):
        """Export a single approved mask to TIFF file"""
        if not self.masks_dir:
            self.log("   ⚠️ No masks directory set - cannot export")
            return

        mask_data = flat_data['mask_data']
        img_name = flat_data['image_name']
        soma_id = mask_data['soma_id']
        area_um2 = mask_data.get('area_um2', 0)

        # Create unique filename with area to distinguish different masks for same soma
        img_basename = os.path.splitext(img_name)[0]
        mask_filename = f"{img_basename}_{soma_id}_area{int(area_um2)}_mask.tif"
        mask_path = os.path.join(self.masks_dir, mask_filename)

        # Get the mask array
        mask = mask_data.get('mask')
        if mask is None:
            self.log(f"   ⚠️ No mask data for {soma_id} - skipping export")
            return

        # Debug: Show mask properties
        self.log(
            f"   🔍 Mask {soma_id}: shape={mask.shape}, dtype={mask.dtype}, min={np.min(mask)}, max={np.max(mask)}, nonzero={np.count_nonzero(mask)}")

        # Check if mask has any data
        if not np.any(mask):
            self.log(f"   ⚠️ Mask {soma_id} is empty (all zeros) - skipping export")
            return

        # Convert mask to 8-bit (0 or 255)
        # Handle both boolean and integer masks
        if mask.dtype == bool:
            mask_8bit = mask.astype(np.uint8) * 255
        else:
            mask_8bit = (mask > 0).astype(np.uint8) * 255

        # Double-check the converted mask has values
        pixels_saved = np.count_nonzero(mask_8bit)
        if pixels_saved == 0:
            self.log(f"   ⚠️ Mask {soma_id} became empty after conversion")
            self.log(f"      Unique values in original: {np.unique(mask)}")
            return

        # Get pixel size
        try:
            pixel_size = float(self.pixel_size_input.text())
        except:
            pixel_size = 0.316  # default

        # Save as TIFF with calibration
        try:
            tifffile.imwrite(
                mask_path,
                mask_8bit,
                resolution=(1.0 / pixel_size, 1.0 / pixel_size),
                metadata={'unit': 'um'}
            )

            self.log(f"   💾 Exported: {mask_filename} ({pixels_saved} pixels)")

        except Exception as e:
            self.log(f"   ❌ Failed to export {mask_filename}: {e}")

    def _export_soma_outline(self, img_name, soma_id, mask, pixel_size, soma_area_um2):
        """Export a soma outline to TIFF file in the somas directory"""
        if not hasattr(self, 'somas_dir') or not self.somas_dir:
            self.log("   ⚠️ No somas directory set - cannot export")
            return

        # Create unique filename matching the mask naming convention
        img_basename = os.path.splitext(img_name)[0]
        soma_filename = f"{img_basename}_{soma_id}_soma.tif"
        soma_path = os.path.join(self.somas_dir, soma_filename)

        # Check if mask has any data
        if not np.any(mask):
            self.log(f"   ⚠️ Soma outline {soma_id} is empty - skipping export")
            return

        # Convert mask to 8-bit (0 or 255)
        if mask.dtype == bool:
            mask_8bit = mask.astype(np.uint8) * 255
        else:
            mask_8bit = (mask > 0).astype(np.uint8) * 255

        # Double-check the converted mask has values
        pixels_saved = np.count_nonzero(mask_8bit)
        if pixels_saved == 0:
            self.log(f"   ⚠️ Soma outline {soma_id} became empty after conversion")
            return

        # Save as TIFF with calibration
        try:
            tifffile.imwrite(
                soma_path,
                mask_8bit,
                resolution=(1.0 / pixel_size, 1.0 / pixel_size),
                metadata={'unit': 'um'}
            )

            self.log(f"   💾 Saved soma: {soma_filename} ({pixels_saved} pixels, {soma_area_um2:.1f} µm²)")

        except Exception as e:
            self.log(f"   ❌ Failed to save {soma_filename}: {e}")

    def _advance_to_next_unreviewed(self):
        """Skip to next unreviewed mask, or complete QA if all done"""
        original_idx = self.mask_qa_idx

        # Search forward for next unreviewed mask
        for idx in range(self.mask_qa_idx + 1, len(self.all_masks_flat)):
            if self.all_masks_flat[idx]['mask_data']['approved'] is None:
                self.mask_qa_idx = idx
                self._show_current_mask()
                return

        # If we get here, check if all masks are reviewed
        all_reviewed = all(flat['mask_data']['approved'] is not None for flat in self.all_masks_flat)

        if all_reviewed:
            self._check_qa_complete()
        else:
            # There are unreviewed masks before current position, stay at last mask
            self.mask_qa_idx = len(self.all_masks_flat) - 1
            self._show_current_mask()
            self.log("⚠️  Reached end. Use Previous to review any remaining masks.")

    def reject_current_mask(self):
        if not self.mask_qa_active or self.mask_qa_idx >= len(self.all_masks_flat):
            return

        flat_data = self.all_masks_flat[self.mask_qa_idx]
        mask_data = flat_data['mask_data']
        mask_data['approved'] = False

        self.log(f"✗ Rejected: {mask_data['soma_id']} ({mask_data['area_um2']} µm²)")

        if self.mask_qa_idx < len(self.all_masks_flat) - 1:
            self.mask_qa_idx += 1
            self._show_current_mask()
        else:
            self._check_qa_complete()

    def next_mask(self):
        if not self.mask_qa_active:
            return
        if self.mask_qa_idx < len(self.all_masks_flat) - 1:
            self.mask_qa_idx += 1
            self._show_current_mask()

    def prev_mask(self):
        if not self.mask_qa_active:
            return
        if self.mask_qa_idx > 0:
            self.mask_qa_idx -= 1
            self._show_current_mask()

    def _check_qa_complete(self):
        all_reviewed = all(flat['mask_data']['approved'] is not None for flat in self.all_masks_flat)

        if all_reviewed:
            self.mask_qa_active = False
            self.approve_mask_btn.setEnabled(False)
            self.reject_mask_btn.setEnabled(False)
            self.prev_btn.setEnabled(False)
            self.next_btn.setEnabled(False)

            # Update image statuses
            for img_name, img_data in self.images.items():
                if img_data['selected'] and img_data['status'] == 'masks_generated':
                    img_data['status'] = 'qa_complete'
                    self._update_file_list_item(img_name)

            self.batch_calculate_btn.setEnabled(True)
            # self.update_workflow_status()

            approved_count = sum(1 for flat in self.all_masks_flat if flat['mask_data']['approved'])
            rejected_count = len(self.all_masks_flat) - approved_count

            self.log("=" * 50)
            self.log(f"✓ QA Complete!")
            self.log(f"Approved: {approved_count}, Rejected: {rejected_count}")
            self.log("=" * 50)

            QMessageBox.information(
                self, "QA Complete",
                f"QA Complete!\n\nApproved: {approved_count}\nRejected: {rejected_count}"
            )

    def batch_calculate_morphology(self):
        try:
            pixel_size = float(self.pixel_size_input.text())

            # No ImageJ required for simple characteristics
            # Complex analysis (Sholl, Skeleton) will be done separately in ImageJ

            approved_masks = [flat for flat in self.all_masks_flat if flat['mask_data']['approved']]
            total = len(approved_masks)

            if total == 0:
                QMessageBox.warning(self, "Warning", "No approved masks to analyze")
                return

            self.log("=" * 50)
            self.log("Calculating simple characteristics (Python)...")
            self.log("Note: Sholl & Skeleton analysis will be done in ImageJ")
            self.log("=" * 50)

            self.progress_bar.setVisible(True)
            self.progress_status_label.setVisible(True)
            self.progress_bar.setValue(0)
            self.start_timer()  # Start the timer

            # Disable the calculate button during processing
            self.batch_calculate_btn.setEnabled(False)

            # Create and start worker thread (no ImageJ needed)
            self.morph_thread = MorphologyCalculationThread(
                approved_masks, pixel_size, use_imagej=False, images=self.images
            )

            # Connect signals
            self.morph_thread.progress.connect(self._on_morph_progress)
            self.morph_thread.finished.connect(self._on_morph_finished)
            self.morph_thread.error_occurred.connect(self._on_morph_error)

            # Start the thread
            self.morph_thread.start()

        except Exception as e:
            self.log(f"Error starting morphology calculation: {e}")
            QMessageBox.critical(self, "Error", f"Failed to start calculation: {e}")
            self.progress_bar.setVisible(False)
            self.progress_status_label.setVisible(False)
            self.stop_timer()

    def _on_morph_progress(self, percentage, status):
        """Update progress bar and status during morphology calculation"""
        self.progress_bar.setValue(percentage)
        self.progress_status_label.setText(status)

    def _on_morph_finished(self, all_results):
        """Handle completion of morphology calculations"""
        self.stop_timer()
        self.progress_bar.setVisible(False)
        self.progress_status_label.setVisible(False)
        self.batch_calculate_btn.setEnabled(True)
        self.timer_label.setVisible(False)

        # Collect metadata for images
        self.log("=" * 50)
        self.log("Checking metadata... (any missing info will be requested)")
        if not self.collect_metadata_for_images():
            self.log("⚠️ Metadata entry cancelled - analysis aborted")
            return

        self._save_batch_results(all_results)

        # Update statuses
        for img_name, img_data in self.images.items():
            if img_data['selected'] and img_data['status'] == 'qa_complete':
                img_data['status'] = 'analyzed'
                self._update_file_list_item(img_name)

        self.log("=" * 50)
        self.log(f"✓ Simple characteristics calculated for {len(all_results)} cells")
        self.log(f"✓ Masks exported to: {self.masks_dir}")
        self.log("")
        self.log("NEXT STEP: Run ImageJ batch analysis")
        self.log("  1. Open Fiji/ImageJ")
        self.log("  2. Run imagej_batch_analysis.ijm macro")
        self.log("  3. Select the masks folder when prompted")
        self.log("=" * 50)

        QMessageBox.information(
            self, "Success",
            f"Simple characteristics calculated for {len(all_results)} cells!\n\n"
            f"Masks exported to:\n{self.masks_dir}\n\n"
            f"NEXT STEP:\n"
            f"Run the ImageJ macro (imagej_batch_analysis.ijm)\n"
            f"to calculate Sholl & Skeleton parameters."
        )

    def _on_morph_error(self, error_msg):
        """Handle errors during morphology calculation"""
        self.stop_timer()
        self.progress_bar.setVisible(False)
        self.progress_status_label.setVisible(False)
        self.timer_label.setVisible(False)
        self.batch_calculate_btn.setEnabled(True)

        self.log(f"ERROR: {error_msg}")
        QMessageBox.critical(self, "Error", f"Morphology calculation failed:\n{error_msg}")

    def open_image_labeling(self):
        """Open the image labeling interface"""
        selected_images = [name for name, data in self.images.items() if data['selected']]

        if not selected_images:
            QMessageBox.warning(self, "No Images Selected",
                                "Please select at least one image first.")
            return

        dialog = QDialog(self)
        dialog.setWindowTitle("Image Labeling")
        dialog.setMinimumWidth(800)
        dialog.setMinimumHeight(600)

        layout = QVBoxLayout(dialog)

        # Instructions
        info_label = QLabel(
            "<b>Enter Animal ID and Treatment for each selected image:</b><br>"
            "You can edit these values anytime before running 'Calculate All Parameters'"
        )
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        # Create table-like layout
        from PyQt5.QtWidgets import QScrollArea, QTableWidget, QTableWidgetItem, QHeaderView

        table = QTableWidget()
        table.setColumnCount(3)
        table.setHorizontalHeaderLabels(["Image Name", "Animal ID", "Treatment"])
        table.setRowCount(len(selected_images))

        # Make first column read-only, stretch to fit
        header = table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(1, QHeaderView.Interactive)
        header.setSectionResizeMode(2, QHeaderView.Interactive)
        table.setColumnWidth(1, 200)
        table.setColumnWidth(2, 200)

        # Populate table
        for row, img_name in enumerate(sorted(selected_images)):
            # Image name (read-only)
            name_item = QTableWidgetItem(img_name)
            name_item.setFlags(name_item.flags() & ~Qt.ItemIsEditable)
            table.setItem(row, 0, name_item)

            # Animal ID (editable)
            animal_item = QTableWidgetItem(self.images[img_name].get('animal_id', ''))
            animal_item.setPlaceholderText = "e.g., Mouse_001"
            table.setItem(row, 1, animal_item)

            # Treatment (editable)
            treatment_item = QTableWidgetItem(self.images[img_name].get('treatment', ''))
            treatment_item.setPlaceholderText = "e.g., Control"
            table.setItem(row, 2, treatment_item)

        layout.addWidget(table)

        # Buttons
        button_layout = QHBoxLayout()

        save_btn = QPushButton("Save")
        save_btn.setDefault(True)
        cancel_btn = QPushButton("Cancel")

        def save_labels():
            # Save all the labels back to the images
            for row in range(table.rowCount()):
                img_name = table.item(row, 0).text()
                animal_id = table.item(row, 1).text().strip()
                treatment = table.item(row, 2).text().strip()

                self.images[img_name]['animal_id'] = animal_id
                self.images[img_name]['treatment'] = treatment

            self.log(f"✓ Labels saved for {table.rowCount()} images")
            dialog.accept()

        save_btn.clicked.connect(save_labels)
        cancel_btn.clicked.connect(dialog.reject)

        button_layout.addStretch()
        button_layout.addWidget(save_btn)
        button_layout.addWidget(cancel_btn)
        layout.addLayout(button_layout)

        dialog.exec_()

    def collect_metadata_for_images(self):
        """Collect AnimalID and Treatment for each processed image"""
        from PyQt5.QtWidgets import QScrollArea

        # Only show dialog for images that don't have metadata yet
        selected_images = [
            name for name, data in self.images.items()
            if data['selected'] and (not data.get('animal_id') or not data.get('treatment'))
        ]

        if not selected_images:
            self.log("✓ All images already have metadata")
            return True

        dialog = QDialog(self)
        dialog.setWindowTitle("Enter Image Metadata")
        dialog.setMinimumWidth(600)
        dialog.setMinimumHeight(400)

        layout = QVBoxLayout(dialog)

        info_label = QLabel("<b>Enter Animal ID and Treatment for each image:</b>")
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        # Scrollable area for multiple images
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)

        # Create input fields for each image
        image_inputs = {}
        for img_name in selected_images:
            img_group = QGroupBox(img_name)
            img_layout = QFormLayout()

            animal_id_input = QLineEdit()
            animal_id_input.setPlaceholderText("e.g., Mouse_001, Rat_A1")
            animal_id_input.setText(self.images[img_name].get('animal_id', ''))

            treatment_input = QLineEdit()
            treatment_input.setPlaceholderText("e.g., Control, Treatment_A, Saline")
            treatment_input.setText(self.images[img_name].get('treatment', ''))

            img_layout.addRow("Animal ID:", animal_id_input)
            img_layout.addRow("Treatment:", treatment_input)

            img_group.setLayout(img_layout)
            scroll_layout.addWidget(img_group)

            image_inputs[img_name] = {
                'animal_id': animal_id_input,
                'treatment': treatment_input
            }

        scroll.setWidget(scroll_content)
        layout.addWidget(scroll)

        # Add buttons
        button_layout = QHBoxLayout()
        ok_btn = QPushButton("OK")
        ok_btn.setDefault(True)
        cancel_btn = QPushButton("Cancel")

        def save_metadata():
            for img_name, inputs in image_inputs.items():
                self.images[img_name]['animal_id'] = inputs['animal_id'].text().strip()
                self.images[img_name]['treatment'] = inputs['treatment'].text().strip()
            dialog.accept()

        ok_btn.clicked.connect(save_metadata)
        cancel_btn.clicked.connect(dialog.reject)

        button_layout.addWidget(ok_btn)
        button_layout.addWidget(cancel_btn)
        layout.addLayout(button_layout)

        return dialog.exec_() == QDialog.Accepted

    def _save_batch_results(self, results):
        if not self.output_dir or not results:
            return

        import csv

        # Add animal_id and treatment to each result
        for result in results:
            img_name_base = result['image_name']
            # Find the matching image data
            for full_name, img_data in self.images.items():
                if os.path.splitext(full_name)[0] == img_name_base:
                    result['animal_id'] = img_data.get('animal_id', '')
                    result['treatment'] = img_data.get('treatment', '')
                    break
            else:
                # If not found, set empty values
                result['animal_id'] = ''
                result['treatment'] = ''

        # Combined results file
        combined_path = os.path.join(self.output_dir, "combined_morphology_results.csv")

        keys = list(results[0].keys())
        # Remove these to reorder them
        for key in ['soma_id', 'image_name', 'animal_id', 'treatment', 'soma_idx']:
            if key in keys:
                keys.remove(key)

        # Put them in the desired order at the front
        ordered_keys = ['image_name', 'animal_id', 'treatment', 'soma_id', 'soma_idx'] + sorted(keys)

        with open(combined_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=ordered_keys)
            writer.writeheader()
            writer.writerows(results)

        self.log(f"Combined results saved to: {combined_path}")

        # Export metadata CSV for ImageJ analysis matching
        metadata_path = os.path.join(self.masks_dir, "mask_metadata.csv")
        with open(metadata_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['mask_filename', 'soma_filename', 'image_name', 'soma_id', 'soma_idx',
                             'soma_x', 'soma_y', 'soma_area_um2', 'cell_area_um2',
                             'pixel_size_um', 'perimeter', 'eccentricity', 'roundness',
                             'cell_spread', 'animal_id', 'treatment'])

            for result in results:
                img_name = result['image_name']
                soma_id = result['soma_id']
                soma_idx = result.get('soma_idx', 0)

                # Get soma position
                soma_x, soma_y = 0, 0
                for full_name, img_data in self.images.items():
                    if os.path.splitext(full_name)[0] == img_name:
                        if soma_idx < len(img_data['somas']):
                            soma_x, soma_y = img_data['somas'][soma_idx]
                        break

                mask_filename = f"{img_name}_{soma_id}_mask.tif"
                soma_filename = f"{img_name}_{soma_id}_soma.tif"

                writer.writerow([
                    mask_filename,
                    soma_filename,
                    img_name,
                    soma_id,
                    soma_idx,
                    f"{soma_x:.2f}",
                    f"{soma_y:.2f}",
                    result.get('soma_area', 0),
                    result.get('area_um2', 0),
                    result.get('mask_area', 0),  # This should be in the results
                    result.get('perimeter', 0),
                    result.get('eccentricity', 0),
                    result.get('roundness', 0),
                    result.get('cell_spread', 0),
                    result.get('animal_id', ''),
                    result.get('treatment', '')
                ])

        self.log(f"Metadata saved to: {metadata_path}")

        # Per-image results files
        by_image = {}
        for result in results:
            img_name = result['image_name']
            if img_name not in by_image:
                by_image[img_name] = []
            by_image[img_name].append(result)

        for img_name, img_results in by_image.items():
            img_path = os.path.join(self.output_dir, f"{img_name}_morphology_results.csv")
            with open(img_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=ordered_keys)
                writer.writeheader()
                writer.writerows(img_results)

        self.log(f"Per-image results saved for {len(by_image)} images")


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = MicrogliaAnalysisGUI()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

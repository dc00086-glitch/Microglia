#!/usr/bin/env python3
"""
MMPS Recovery QA Tool

Recovers a lost MMPS session from files on disk (processed images, soma
outlines, masks) and lets you QA masks immediately. After QA, generates
a .mmps_session file so you can continue in MMPS for morphology calculation.

Usage:
    python recover_qa.py

    Then select your output folder when prompted. The tool will auto-detect
    the masks and somas subfolders.
"""

import sys
import os
import re
import json
import numpy as np
import tifffile
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QMessageBox, QProgressBar
)
from PyQt5.QtGui import QImage, QPixmap, QPainter, QColor, QKeyEvent
from PyQt5.QtCore import Qt


class MaskQAWindow(QMainWindow):
    def __init__(self, output_dir):
        super().__init__()
        self.output_dir = output_dir
        self.masks_dir = os.path.join(output_dir, "masks")
        self.somas_dir = os.path.join(output_dir, "somas")
        self.rejected_dir = os.path.join(self.masks_dir, "rejected")

        self.mask_entries = []  # list of dicts with all info
        self.current_idx = 0

        self.setWindowTitle("MMPS Recovery - Mask QA")
        self.setMinimumSize(900, 700)

        self._scan_folders()
        if not self.mask_entries:
            QMessageBox.critical(self, "No Masks Found",
                f"No mask TIFFs found in:\n{self.masks_dir}")
            sys.exit(1)

        self._init_ui()
        self._show_current()

    def _scan_folders(self):
        """Scan folders and build the list of masks to QA."""
        if not os.path.isdir(self.masks_dir):
            return

        # Find all processed images
        processed_images = {}
        for f in sorted(os.listdir(self.output_dir)):
            if f.endswith("_processed.tif"):
                name = f.replace("_processed.tif", "")
                processed_images[name] = os.path.join(self.output_dir, f)

        # Find all mask files
        mask_pattern = re.compile(r'^(.+?)_(soma_\d+_\d+)_area(\d+)_mask\.tif$')
        for f in sorted(os.listdir(self.masks_dir)):
            m = mask_pattern.match(f)
            if not m:
                continue

            img_name = m.group(1)
            soma_id = m.group(2)
            area = int(m.group(3))

            # Find the matching processed image
            proc_path = processed_images.get(img_name)
            if not proc_path:
                continue

            mask_path = os.path.join(self.masks_dir, f)

            self.mask_entries.append({
                'filename': f,
                'mask_path': mask_path,
                'processed_path': proc_path,
                'img_name': img_name,
                'soma_id': soma_id,
                'area_um2': area,
                'approved': None,  # None=unreviewed, True=approved, False=rejected
            })

        # Sort by image name, then soma_id, then area (largest first)
        self.mask_entries.sort(key=lambda e: (e['img_name'], e['soma_id'], -e['area_um2']))

        print(f"Found {len(self.mask_entries)} masks across {len(processed_images)} images")

    def _init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # Status bar
        self.status_label = QLabel("")
        self.status_label.setStyleSheet("font-size: 14px; font-weight: bold; padding: 5px;")
        layout.addWidget(self.status_label)

        # Image display
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumHeight(500)
        self.image_label.setStyleSheet("background-color: black;")
        layout.addWidget(self.image_label, stretch=1)

        # Buttons
        btn_layout = QHBoxLayout()

        self.prev_btn = QPushButton("< Previous (Left)")
        self.prev_btn.clicked.connect(self.prev_mask)
        btn_layout.addWidget(self.prev_btn)

        self.reject_btn = QPushButton("Reject (R)")
        self.reject_btn.clicked.connect(self.reject_current)
        self.reject_btn.setStyleSheet("border: 2px solid #F44336; font-weight: bold; padding: 8px;")
        btn_layout.addWidget(self.reject_btn)

        self.approve_btn = QPushButton("Approve (A)")
        self.approve_btn.clicked.connect(self.approve_current)
        self.approve_btn.setStyleSheet("border: 2px solid #4CAF50; font-weight: bold; padding: 8px;")
        btn_layout.addWidget(self.approve_btn)

        self.next_btn = QPushButton("Next > (Right)")
        self.next_btn.clicked.connect(self.next_mask)
        btn_layout.addWidget(self.next_btn)

        layout.addLayout(btn_layout)

        # Progress
        self.progress = QProgressBar()
        self.progress.setMaximum(len(self.mask_entries))
        layout.addWidget(self.progress)

        # Finish button
        self.finish_btn = QPushButton("Finish QA & Generate Session File")
        self.finish_btn.clicked.connect(self.finish_qa)
        self.finish_btn.setStyleSheet("border: 2px solid #2196F3; font-weight: bold; padding: 10px;")
        layout.addWidget(self.finish_btn)

    def _show_current(self):
        if self.current_idx >= len(self.mask_entries):
            return

        entry = self.mask_entries[self.current_idx]

        # Load processed image
        proc_img = tifffile.imread(entry['processed_path'])
        # Normalize to 8-bit for display
        if proc_img.dtype != np.uint8:
            pmin, pmax = proc_img.min(), proc_img.max()
            if pmax > pmin:
                proc_img = ((proc_img - pmin) / (pmax - pmin) * 255).astype(np.uint8)
            else:
                proc_img = np.zeros_like(proc_img, dtype=np.uint8)

        # Load mask
        mask = tifffile.imread(entry['mask_path'])
        mask_binary = mask > 0

        # Create RGB overlay
        h, w = proc_img.shape[:2]
        rgb = np.stack([proc_img, proc_img, proc_img], axis=2).copy()

        # Green overlay where mask is
        overlay_alpha = 0.4
        rgb[mask_binary, 1] = np.clip(
            rgb[mask_binary, 1] * (1 - overlay_alpha) + 255 * overlay_alpha, 0, 255
        ).astype(np.uint8)

        # Convert to QPixmap
        qimg = QImage(rgb.data, w, h, w * 3, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)

        # Scale to fit label
        label_size = self.image_label.size()
        scaled = pixmap.scaled(label_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_label.setPixmap(scaled)

        # Update status
        status = entry['approved']
        status_text = "APPROVED" if status is True else "REJECTED" if status is False else "Unreviewed"
        n_reviewed = sum(1 for e in self.mask_entries if e['approved'] is not None)

        self.status_label.setText(
            f"Mask {self.current_idx + 1}/{len(self.mask_entries)} | "
            f"{entry['img_name']} | {entry['soma_id']} | "
            f"Area: {entry['area_um2']} um² | {status_text} | "
            f"Reviewed: {n_reviewed}/{len(self.mask_entries)}"
        )

        self.progress.setValue(n_reviewed)

    def approve_current(self):
        if self.current_idx < len(self.mask_entries):
            entry = self.mask_entries[self.current_idx]
            entry['approved'] = True

            # Auto-approve all smaller masks for the same soma
            img_name = entry['img_name']
            soma_id = entry['soma_id']
            approved_area = entry['area_um2']
            auto_count = 0
            for other in self.mask_entries:
                if (other is not entry
                        and other['img_name'] == img_name
                        and other['soma_id'] == soma_id
                        and other['area_um2'] < approved_area
                        and other['approved'] is None):
                    other['approved'] = True
                    auto_count += 1

            if auto_count > 0:
                print(f"Auto-approved {auto_count} smaller mask(s) for {img_name} {soma_id}")

            self._advance()

    def reject_current(self):
        if self.current_idx < len(self.mask_entries):
            entry = self.mask_entries[self.current_idx]
            entry['approved'] = False

            # Auto-reject all smaller masks for the same soma
            img_name = entry['img_name']
            soma_id = entry['soma_id']
            rejected_area = entry['area_um2']
            auto_count = 0
            for other in self.mask_entries:
                if (other is not entry
                        and other['img_name'] == img_name
                        and other['soma_id'] == soma_id
                        and other['area_um2'] < rejected_area
                        and other['approved'] is None):
                    other['approved'] = False
                    auto_count += 1

            if auto_count > 0:
                print(f"Auto-rejected {auto_count} smaller mask(s) for {img_name} {soma_id}")

            self._advance()

    def _advance(self):
        """Move to next unreviewed mask."""
        for i in range(self.current_idx + 1, len(self.mask_entries)):
            if self.mask_entries[i]['approved'] is None:
                self.current_idx = i
                self._show_current()
                return

        # All reviewed - check
        all_reviewed = all(e['approved'] is not None for e in self.mask_entries)
        if all_reviewed:
            n_approved = sum(1 for e in self.mask_entries if e['approved'])
            n_rejected = len(self.mask_entries) - n_approved
            self._show_current()
            QMessageBox.information(self, "QA Complete",
                f"All masks reviewed!\n\n"
                f"Approved: {n_approved}\nRejected: {n_rejected}\n\n"
                f"Click 'Finish QA & Generate Session File' to save.")
        else:
            # Stay at last
            self.current_idx = len(self.mask_entries) - 1
            self._show_current()

    def next_mask(self):
        if self.current_idx < len(self.mask_entries) - 1:
            self.current_idx += 1
            self._show_current()

    def prev_mask(self):
        if self.current_idx > 0:
            self.current_idx -= 1
            self._show_current()

    def keyPressEvent(self, event):
        key = event.key()
        if key == Qt.Key_A:
            self.approve_current()
        elif key == Qt.Key_R:
            self.reject_current()
        elif key == Qt.Key_Left:
            self.prev_mask()
        elif key == Qt.Key_Right:
            self.next_mask()
        elif key == Qt.Key_Space:
            self.approve_current()
        else:
            super().keyPressEvent(event)

    def finish_qa(self):
        """Move rejected masks to rejected/ folder and generate session file."""
        n_approved = sum(1 for e in self.mask_entries if e['approved'] is True)
        n_rejected = sum(1 for e in self.mask_entries if e['approved'] is False)
        n_unreviewed = sum(1 for e in self.mask_entries if e['approved'] is None)

        if n_unreviewed > 0:
            reply = QMessageBox.question(self, "Unreviewed Masks",
                f"{n_unreviewed} masks have not been reviewed.\n\n"
                f"Unreviewed masks will be kept as approved.\n\nContinue?",
                QMessageBox.Yes | QMessageBox.No)
            if reply != QMessageBox.Yes:
                return

        # Move rejected masks to rejected/ subfolder
        if n_rejected > 0:
            os.makedirs(self.rejected_dir, exist_ok=True)

        for entry in self.mask_entries:
            if entry['approved'] is False:
                src = entry['mask_path']
                dst = os.path.join(self.rejected_dir, entry['filename'])
                if os.path.exists(src):
                    os.rename(src, dst)

        # Generate session file for MMPS
        self._generate_session_file()

        QMessageBox.information(self, "Done",
            f"QA Complete!\n\n"
            f"Approved: {n_approved}\n"
            f"Rejected: {n_rejected} (moved to {self.rejected_dir})\n"
            f"Unreviewed (kept): {n_unreviewed}\n\n"
            f"Session file saved to:\n"
            f"{os.path.join(self.output_dir, 'recovered.mmps_session')}\n\n"
            f"Load this in MMPS to continue with morphology calculation.")

        self.close()

    def _generate_session_file(self):
        """Generate a .mmps_session file from the disk state."""
        # Group by image
        images_data = {}
        for entry in self.mask_entries:
            img_name = entry['img_name']
            if img_name not in images_data:
                images_data[img_name] = {
                    'processed_path': entry['processed_path'],
                    'somas': {},  # soma_id -> centroid
                }
            images_data[img_name]['somas'][entry['soma_id']] = True

        # Build session
        session = {
            'version': 2,
            'output_dir': self.output_dir,
            'masks_dir': self.masks_dir,
            'colocalization_mode': False,
            'pixel_size': '0.316',
            'rolling_ball_radius': 50,
            'use_min_intensity': True,
            'min_intensity_percent': 30,
            'mask_min_area': 300,
            'mask_max_area': 800,
            'mask_step_size': 120,
            'coloc_channel_1': 0,
            'coloc_channel_2': 1,
            'grayscale_channel': 0,
            'images': {}
        }

        # Try to find original raw images by looking for non-processed TIFFs
        # in a parent directory or the user can fix paths later
        for img_name, data in images_data.items():
            # Extract soma centroids from soma_id format: soma_ROW_COL
            somas = []
            soma_ids = []
            soma_outlines = []

            for soma_id in sorted(data['somas'].keys()):
                # Parse centroid from soma_id
                match = re.match(r'soma_(\d+)_(\d+)', soma_id)
                if match:
                    row, col = int(match.group(1)), int(match.group(2))
                    somas.append([float(row), float(col)])
                    soma_ids.append(soma_id)

                    # Try to load soma outline from somas folder
                    soma_file = os.path.join(self.somas_dir, f"{img_name}_{soma_id}_soma.tif")
                    if os.path.exists(soma_file):
                        soma_mask = tifffile.imread(soma_file)
                        soma_coords = np.argwhere(soma_mask > 0)
                        if len(soma_coords) > 0:
                            # Calculate soma area
                            pixel_size = 0.316  # default
                            soma_area = len(soma_coords) * (pixel_size ** 2)
                            # Get centroid from mask (more accurate than filename)
                            center_row = float(np.mean(soma_coords[:, 0]))
                            center_col = float(np.mean(soma_coords[:, 1]))
                            somas[-1] = [center_row, center_col]

                            # Build outline data
                            soma_outlines.append({
                                'soma_idx': len(soma_outlines),
                                'soma_id': soma_id,
                                'centroid': [center_row, center_col],
                                'soma_area_um2': soma_area,
                                'polygon_points': [],  # Can't recover polygon from binary mask
                            })
                    else:
                        # No soma file - use filename centroid
                        soma_outlines.append({
                            'soma_idx': len(soma_outlines),
                            'soma_id': soma_id,
                            'centroid': [float(row), float(col)],
                            'soma_area_um2': 0,
                            'polygon_points': [],
                        })

            # Find the raw image path - check common locations
            raw_path = data['processed_path']  # Fallback to processed
            possible_raw = os.path.join(os.path.dirname(self.output_dir), img_name + ".tif")
            if os.path.exists(possible_raw):
                raw_path = possible_raw

            # Count remaining mask files for this image
            mask_files = [
                f for f in os.listdir(self.masks_dir)
                if f.startswith(img_name + "_") and f.endswith('_mask.tif')
            ]

            session['images'][img_name + ".tif"] = {
                'raw_path': raw_path,
                'processed_path': data['processed_path'],
                'status': 'qa_complete',
                'selected': True,
                'animal_id': '',
                'treatment': '',
                'rolling_ball_radius': 50,
                'somas': somas,
                'soma_ids': soma_ids,
                'soma_outlines': soma_outlines,
                'mask_files': mask_files,
            }

        path = os.path.join(self.output_dir, "recovered.mmps_session")
        with open(path, 'w') as f:
            json.dump(session, f, indent=2)

        print(f"Session file saved to: {path}")


def main():
    app = QApplication(sys.argv)

    # Ask for output folder
    output_dir = QFileDialog.getExistingDirectory(
        None, "Select your MMPS Output Folder",
        options=QFileDialog.DontUseNativeDialog
    )
    if not output_dir:
        print("No folder selected.")
        sys.exit(0)

    masks_dir = os.path.join(output_dir, "masks")
    if not os.path.isdir(masks_dir):
        # Maybe they selected the parent - check for common structure
        QMessageBox.critical(None, "Error",
            f"No 'masks' subfolder found in:\n{output_dir}\n\n"
            f"Please select the output folder that contains the 'masks' subfolder.")
        sys.exit(1)

    window = MaskQAWindow(output_dir)
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()

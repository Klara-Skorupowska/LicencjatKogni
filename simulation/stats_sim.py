### all simulation environment statistics are collected here

import numpy as np
import cv2

import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout, QGroupBox
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt

import os
import csv
from datetime import datetime

class StatsSim():
    def __init__(self):
        pass
    def update(self):
        pass
    def close(self):
        pass


### subclasses for collecting and/or displaying 
class StatusBoard(StatsSim, QWidget):
    def __init__(self, name, obj):
        # 1. Initialize the Qt Application if it hasn't been started
        self.app = QApplication.instance()
        if self.app is None:
            self.app = QApplication(sys.argv)
            
        StatsSim.__init__(self)
        QWidget.__init__(self)
        self.setWindowTitle(f"{name} Pilot View")
        self.robot = obj
        
        # Safely extract dictionaries
        self.sensors = getattr(obj, 'sensors', {})
        self.actuators = getattr(obj, 'actuators', {})

        if not self.sensors and not self.actuators:
            raise ValueError("Status Board fail: no data to display")

        # 2. Safely locate the camera 
        self.camera = None
        for sensor in self.sensors.values():
            if type(sensor).__name__ == 'CameraSensor':
                self.camera = sensor
                break

        # 3. Build the UI Layout
        main_layout = QVBoxLayout()

        # HUD / Text Panel
        self.hud_group = QGroupBox("Telemetry")
        hud_layout = QVBoxLayout()
        self.text_label = QLabel("Waiting for data...")
        self.text_label.setWordWrap(True) 
        self.text_label.setStyleSheet("font-family: monospace; font-size: 12pt;")
        hud_layout.addWidget(self.text_label)
        self.hud_group.setLayout(hud_layout)
        main_layout.addWidget(self.hud_group)

        # Video Panel
        if self.camera:
            self.video_label = QLabel("No signal")
            self.video_label.setAlignment(Qt.AlignCenter)
            self.video_label.setStyleSheet("background-color: black; color: white;")
            main_layout.addWidget(self.video_label)

        self.setLayout(main_layout)
        
        # Set minimum width so the HUD text has room to breathe
        self.setMinimumWidth(800)
        self.show()

    def update(self):
        """Called by your external main loop to fetch data and refresh the UI."""
        
        # 1. Update Text HUD
        hud_lines = ["<b>Sensors:</b>"]
        for name, sensor in self.sensors.items():
            if sensor is self.camera:
                continue
                
            data = sensor.get_data()
            hud_lines.append(self._format_data(name, data))

        hud_lines.append("<br><b>Actuators:</b>")
        for name, actuator in self.actuators.items():
            data = actuator.get_value()
            hud_lines.append(self._format_data(name, data))

        # Join the lines with HTML breaks and update the label
        self.text_label.setText("<br>".join(hud_lines))

        # 2. Update Video Feed
        if self.camera:
            frame = self.camera.get_data()
            if frame is not None:
                # Convert from OpenCV (BGR/BGRA) to PyQt (RGB)
                if len(frame.shape) == 3:
                    if frame.shape[2] == 4:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
                    else:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                h, w, ch = frame.shape
                bytes_per_line = ch * w
                
                # Create QImage and display it
                q_img = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
                self.video_label.setPixmap(QPixmap.fromImage(q_img))

        # 3. Process events to keep UI responsive
        # This replaces cv2.waitKey(1)
        self.app.processEvents()

    def _format_data(self, name, data):
        """Helper method to cleanly format single values or arrays of values."""
        if isinstance(data, (list, tuple, np.ndarray)):
            # Format lists (e.g. from SensorArray)
            string_data = ', '.join(f"{d:.2f}" for d in data if isinstance(d, (int, float)))
        elif isinstance(data, (int, float)):
            # Format single floats
            string_data = f"{data:.2f}"
        else:
            # Fallback for strings, booleans, or None
            string_data = str(data)
            
        return f"&nbsp;&nbsp;{name}: {string_data}"


class DataLogger(StatsSim):
    """
    Collects sensor and actuator data (excluding cameras) and writes it to a CSV file.
    Creates flattened headers for array-based sensors (e.g., lidars -> lidar_0, lidar_1).
    """
    def __init__(self, obj, log_dir="telemetry"):
        self.robot = obj
        self.sensors = getattr(obj, 'sensors', {})
        self.actuators = getattr(obj, 'actuators', {})

        start_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = f"logs/{start_time}/{log_dir}"
        
        # Ensure the log directory exists
        os.makedirs(log_dir, exist_ok=True)
        name = self.robot.model_name
        self.filepath = os.path.join(log_dir, f"telemetry_{name}.csv")
        
        # Initialize headers
        self.headers = ["timestamp"]
        self._build_headers()
        
        # Keep the file open for fast, continuous writing during the simulation loop
        self.file = open(self.filepath, mode='w', newline='')
        self.writer = csv.writer(self.file)
        self.writer.writerow(self.headers)
        
        print(f"DataLogger initialized. Logging to: {self.filepath}")

    def _build_headers(self):
        """Samples the data once at startup to create accurate CSV columns."""
        # 1. Map Sensor columns
        for name, sensor in self.sensors.items():
            if type(sensor).__name__ == 'CameraSensor':
                continue
                
            data = sensor.get_data()
            if isinstance(data, (list, tuple)):
                # If it's a SensorArray returning a list, create a column for each item
                for i in range(len(data)):
                    self.headers.append(f"{name}_{i}")
            else:
                self.headers.append(name)

        # 2. Map Actuator columns
        for name, actuator in self.actuators.items():
            data = actuator.get_value()
            if isinstance(data, (list, tuple)):
                for i in range(len(data)):
                    self.headers.append(f"{name}_{i}")
            else:
                self.headers.append(name)

    def update(self):
        """Fetches current data and writes a single row to the CSV."""
        # Get current time down to the microsecond
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
        row = [current_time]
        
        # 1. Fetch Sensor data
        for name, sensor in self.sensors.items():
            if type(sensor).__name__ == 'CameraSensor':
                continue
                
            data = sensor.get_data()
            self._append_data(row, data)
            
        # 2. Fetch Actuator data
        for name, actuator in self.actuators.items():
            data = actuator.get_value()
            self._append_data(row, data)
            
        # Write the row immediately
        self.writer.writerow(row)

    def _append_data(self, row, data):
        """Helper to flatten lists into the CSV row."""
        if isinstance(data, (list, tuple)):
            row.extend(data)
        else:
            row.append(data)

    def close(self):
        """Closes the file. Call this when shutting down your simulation!"""
        if not self.file.closed:
            self.file.close()
            print("DataLogger safely closed.")
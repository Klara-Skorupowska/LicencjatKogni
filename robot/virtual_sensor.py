# a parent class for virtual sensors

import cv2
import numpy as np

from communicator import Communicator


class VirtualSensor():
    def __init__(self, bus: Communicator):
        self.bus = bus

    def read(self): # from the bus
        raise NotImplementedError("The read() method must be implemented in the subclass.")

class VirtualSensorArray(VirtualSensor):
    def __init__(self, bus: Communicator, sensors: list[VirtualSensor]):
        super().__init__(bus)
        self.sensors = sensors
        self.value = []
    def read(self):
        self.value = [sensor.read() for sensor in self.sensors]
        return self.value

class LidarSensor(VirtualSensor):
    def __init__(self, bus: Communicator, lidar_direction):
        '''
        lidar_direction: - in degrees, 0 is forward, 90 is left, 180 is backward, 270 is right
        '''
        super().__init__(bus)
        self.lidar_direction = lidar_direction
        self.value = None

    def read(self):
        self.value = self.bus.call_service(f"/sensor/lidar_{self.lidar_direction}/sense")
        return self.value

class CameraSensor(VirtualSensor):
    def __init__(self, bus: Communicator):
        '''
        res (resolution) - [heigth, width, channels]
        '''
        super().__init__(bus)
        self.frame = None # raw frame from the camera
        self.coded = None # information from the frame: colors and edges

    def read(self):
        self.frame = self.bus.call_service("/sensor/camera/sense")
        return self.frame

    def show_hue_debug_grid(self, mean_hues, original_shape):
        """
        Visualizes the 3x3 grid of mean hues.
    
        mean_hues: List of 9 floats (0.0 to 1.0)
        original_shape: The shape of the original camera frame (e.g., frame.shape)
        """
        h, w = original_shape[:2]
        cell_h = h // 3
        cell_w = w // 3
    
        # 1. Create a blank image in HSV color space
        debug_hsv = np.zeros((h, w, 3), dtype=np.uint8)
    
        # 2. Set Saturation and Value to maximum (255) for pure, bright colors
        debug_hsv[:, :, 1] = 255
        debug_hsv[:, :, 2] = 255
    
        # 3. Fill the grid cells with the calculated hues
        for i in range(3):
            for j in range(3):
                # Extract the correct hue from the flat list (0 to 8)
                idx = i * 3 + j
                norm_hue = mean_hues[idx]
            
                # Convert normalized hue (0.0-1.0) back to OpenCV's (0-179) scale
                cv_hue = int(norm_hue * 179)
            
                # Calculate pixel boundaries, ensuring we cover the edges perfectly
                y_start = i * cell_h
                y_end = (i + 1) * cell_h if i < 2 else h
                x_start = j * cell_w
                x_end = (j + 1) * cell_w if j < 2 else w
            
                # Apply the hue to the H channel (index 0) of this specific cell
                debug_hsv[y_start:y_end, x_start:x_end, 0] = cv_hue
            
        # 4. Convert back to BGR so OpenCV can display it on your screen
        debug_bgr = cv2.cvtColor(debug_hsv, cv2.COLOR_HSV2BGR)
    
        # 5. Draw black grid lines so you can easily see the cell boundaries
        for i in range(1, 3):
            cv2.line(debug_bgr, (0, i * cell_h), (w, i * cell_h), (0, 0, 0), 2)
            cv2.line(debug_bgr, (i * cell_w, 0), (i * cell_w, h), (0, 0, 0), 2)
        
        # 6. Show the frame
        cv2.imshow("Mean Hues Debug", debug_bgr)
        cv2.waitKey(1)

    def preprocess(self):
        '''
        preprocess the frame for the neural network: returns 9 colors (RGB) and 4 edge counts (vertical , horizontal, diagonal left, diagonal right)
        '''
        if self.frame is None:
            raise ValueError("No frame available. Please call read() before preprocess().")

        # calculate the average color (hue) of the frame devided into 3x3 grid
        # 1. Convert to HSV and extract ONLY the Hue channel
        hsv_frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2HSV)
        hue_channel = hsv_frame[:, :, 0] # OpenCV stores Hue as 0-179 (half of 360)
        
        # Pre-calculate cell dimensions for speed
        h, w = hue_channel.shape
        cell_h = h // 3
        cell_w = w // 3
        
        mean_hues = []
        for i in range(3):
            for j in range(3):
                # 2. Slice the 3x3 grid
                grid = hue_channel[i * cell_h : (i + 1) * cell_h, j * cell_w : (j + 1) * cell_w]
                
                # 3. Convert OpenCV Hue (0-179) to radians (0 to 2*pi)
                # Multiply by 2.0 to restore the 360 scale, then convert to radians
                hue_rad = np.deg2rad(grid.astype(float) * 2.0)
                
                # 4. Calculate Circular Mean using Cosine and Sine
                mean_x = np.mean(np.cos(hue_rad))
                mean_y = np.mean(np.sin(hue_rad))
                
                # Convert back to an angle (-pi to +pi)
                mean_angle_rad = np.arctan2(mean_y, mean_x)
                
                # Convert back to degrees (0 to 360)
                mean_angle_deg = np.rad2deg(mean_angle_rad)
                if mean_angle_deg < 0:
                    mean_angle_deg += 360.0
                    
                # 5. Normalize to 0.0 - 1.0 for the GNG State Array
                mean_hues.append(float(mean_angle_deg / 360.0))

        # calculate edge counts using cv2
        # 1. Blur to remove minor noise before edge detection
        blurred = cv2.GaussianBlur(self.frame, (5, 5), 0)
        gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)

        # 2. Extract line segments
        # rho=1, theta=pi/180, threshold=50, minLineLength=30, maxLineGap=10
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=30, minLineLength=30, maxLineGap=10)

        ### --- DEBUGGING VISUALIZATION ---
        # add them in pink to the blurred image for debugging
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(blurred, (x1, y1), (x2, y2), (255, 0, 255), 2)

        cv2.imshow("Blurred", blurred)
        cv2.imshow("Edges", edges)
        self.show_hue_debug_grid(mean_hues, self.frame.shape)
        cv2.waitKey(1)
        ### --- DEBUGGING VISUALIZATION END ---

        # 3. Initialize counts for your 4 target variables
        vertical_count = 0
        horizontal_count = 0
        diag_45_count = 0
        diag_135_count = 0

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
        
                # Calculate the angle of the line (-180 to 180)
                angle = np.arctan2(y2 - y1, x2 - x1) * 180.0 / np.pi
        
                # Normalize the angle to be strictly between 0 and 180 degrees
                if angle < 0:
                    angle += 180.0
            
                # 4. Bucket the line into the correct category based on its angle
                if angle < 10 or angle > 170:
                    horizontal_count += 1
                elif 80 < angle < 100:
                    vertical_count += 1
                elif 35 <= angle <= 55:
                    diag_45_count += 1    # The '\' direction
                elif 125 <= angle <= 145:
                    diag_135_count += 1   # The '/' direction

        # The 4 ints you need for your state array
        edge_counts = [vertical_count, horizontal_count, diag_45_count, diag_135_count]

        self.coded = mean_hues + edge_counts

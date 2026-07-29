### all simulation environment statistics are collected here

import numpy as np
import cv2

class Stats():
    def __init__(self):
        pass
    def update(self):
        pass
    def show(self):
        pass


### subclass for collecting and displaying online statistics

class StatusBoard(Stats):
    '''
    class with live feed of the robot's sensors data and actuators parameters, and a video feed (if available)
    '''
    def __init__(self, id: str, sensors: dict, actuators: dict, video = None):
        '''
        init the status board. It is a floating panel.
            sensors: the sensors to display in a dictionary form (name - value (float)) 
            actuators: the actuators to display, dictionary form (name - value (float))
            video: the video feed to display, if available
        '''
        super().__init__()
        self.id = id
        self.sensors = sensors
        len_sensors = len(sensors) if sensors else 0
        self.actuators = actuators
        len_actuators = len(actuators) if actuators else 0
        self.video = video

        if self.video != None:
            _, video_width, _= self.video.shape
            n = 1
        else:
            video_width = 0
            n = 0

        column_width = 160
        panel_height = 80 
        panel_width = max(video_width, (max(len_sensors - n, len_actuators))*column_width+20)
        self.panel = np.zeros((panel_height, panel_width, 3), dtype=np.uint8)


    def update(self, sensors: dict, actuators: dict, video = None):
        self.sensors = sensors
        self.actuators = actuators
        self.video = video

    def show(self):
        self.panel.fill(0) # clear the panel

        if self.video != None:
            # convert RGBA if necessary
            if self.video.shape[2] == 4:
                frame = cv2.cvtColor(self.video, cv2.COLOR_RGBA2BGR)
            if frame.shape[1] != self.width:
                frame = cv2.resize(frame, (self.width, int(frame.shape[0] * (self.width / frame.shape[1]))))
        else: 
            frame = np.zeros((1, self.width, 3), dtype=np.uint8)

        # Add the HUD Text
        font = cv2.FONT_HERSHEY_SIMPLEX
        color = (255, 255, 255)

        def draw_dict(data, y_pos):
            x_offset = 10
            
            for key, value in data.items():
                if value.__class__ != float: continue # skip non-float values 
                
                text = f"{key}: {value:.2f}"
                cv2.putText(self.panel, text, (x_offset, y_pos), font, 0.5, color, 1, cv2.LINE_AA)
                
                x_offset += self.column_width
                # Wrap text to next line if it exceeds width
                if x_offset > self.width - self.column_width:
                    x_offset = 10
                    y_pos += 20 
            return y_pos

        # Draw Observations on the first line(s)
        current_y = draw_dict(self.sensors, 25)
        
        # Draw Actions on the next line(s)
        draw_dict(self.actuators, current_y + 25)

        # Show the combined view 
        combined_view = np.vstack((self.panel, frame))
        cv2.imshow(f"{self.id}: Pilot View", combined_view)
        cv2.waitKey(1)

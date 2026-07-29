import numpy as np
import cv2

class StatusBoard:
    def __init__(self, name, observations, actions):
        '''
            init the status board. It will be a panel below the main view, showing the observations, actions and colors in text form
            iputs: 
                name: the name of the robot, will be shown in the window title
                observations: the observations to display in a dictionary form (name - value)
                actions: the actions to display, dictionary form (name - value)
        '''
        if "video" in observations:
            frame = observations["video"]
            _, fw, _= frame.shape
            n = 1
        else:
            fw = 0
            n = 0

        self.column_width = 160
        self.panel_h = 80 
        self.width = max(fw, (max(len(observations) - n,len(actions)))*self.column_width+20) 
        self.panel = np.zeros((self.panel_h, self.width, 3), dtype=np.uint8)
        self.name = name

    def update(self, observations, actions):

        self.panel.fill(0)

        if "video" in observations:
            frame = observations["video"]
            # convert RGBA if necessary
            if frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
            if frame.shape[1] != self.width:
                frame = cv2.resize(frame, (self.width, int(frame.shape[0] * (self.width / frame.shape[1]))))
        else: 
            frame = np.zeros((1, self.width, 3), dtype=np.uint8)

        # Add the HUD Text
        font = cv2.FONT_HERSHEY_SIMPLEX
        color = (255, 255, 255)

        def draw_dict(data, y_pos, label_prefix=""):
            x_offset = 10
            
            for key, value in data.items():
                if key == "video": continue
                
                text = f"{key}: {value:.2f}"
                cv2.putText(self.panel, text, (x_offset, y_pos), font, 0.5, color, 1, cv2.LINE_AA)
                
                x_offset += self.column_width
                # Wrap text to next line if it exceeds width
                if x_offset > self.width - self.column_width:
                    x_offset = 10
                    y_pos += 20 
            return y_pos

        # Draw Observations on the first line(s)
        current_y = draw_dict(observations, 25)
        
        # Draw Actions on the next line(s)
        draw_dict(actions, current_y + 25)

        # Show the combined view 
        combined_view = np.vstack((self.panel, frame))
        cv2.imshow(f"{self.name} Pilot View", combined_view)
        cv2.waitKey(1) 
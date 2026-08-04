

class Skill():
    '''
    predefined behavior that can be used by the robot. It works if certain conditions are met and it changes the world in a specific way.
    '''
    
    def update(self, observations):
        """
        Takes current observations map and returns actuators map and is_done_flag)
        """
        raise NotImplementedError("Subclasses must implement this method")

class MoveForward(Skill):
    '''
    A simple skill that moves the robot forward until it detects an obstacle in front.
    '''
    def __init__(self, speed=1.0, threshold=1.0):
        self.speed = speed
        self.threshold = threshold
        self.is_done = False

    def update(self, observations):
        front_dist = observations.get("dist_front", float('inf'))
        if front_dist < self.threshold:
            self.is_done = True
            return {"left_motor": 0.0, "right_motor": 0.0}, self.is_done
        else:
            return {"left_motor": self.speed, "right_motor": self.speed}, self.is_done

class Turn(Skill):
    '''
    A simple skill that turns the robot until it detects no obstacle in front.
    '''
    def __init__(self, speed=0.5, threshold=1.0):
        self.speed = speed
        self.threshold = threshold
        self.is_done = False
        
        self.push_steps = 0
        self.direction = None

    def update(self, observations):
        front_dist = observations.get("dist_front", float('inf'))
        self.push_steps += 1

        if front_dist >= self.threshold:
            self.is_done = True
            self.direction = None
            return {"left_motor": 0.0, "right_motor": 0.0}, self.is_done
        else:
            # dont change direction too often, to avoid jittering
            if self.direction is not None and self.push_steps < 20: 
                if self.direction == "right":
                    return {"left_motor": self.speed, "right_motor": -self.speed}, self.is_done
                elif self.direction == "left":
                    return {"left_motor": -self.speed, "right_motor": self.speed}, self.is_done

            # decide direction based on which side is more clear
            if observations["dist_left"] < observations["dist_right"]:
                self.direction = "right"
                return {"left_motor": self.speed, "right_motor": -self.speed}, self.is_done
            else:
                self.direction = "left"
                return {"left_motor": -self.speed, "right_motor": self.speed}, self.is_done
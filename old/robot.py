class Robot:
    ''' Class for robot. It bridges brain and body.'''
    def check_compatibility(self, brain, body):
        ''' Checks if brain and body are compatible, i.e. if brain requires observations and actuators that body provides.'''
        brain.obsrv_keys = set(brain.observations_map.keys())
        body.obsrv_keys = set(body.observations_map.keys())
        if not brain.obsrv_keys.issubset(body.obsrv_keys):
            raise ValueError(f"Brain requires observations {brain.obsrv_keys} that body does not provide {body.obsrv_keys}")
        
        brain.actuator_keys = set(brain.actuators_map.keys())
        body.actuator_keys = set(body.actuators_map.keys())
        if not brain.actuator_keys.issubset(body.actuator_keys):
            raise ValueError(f"Brain requires actuators {brain.actuator_keys} that body does not provide {body.actuator_keys}")
        

    def __init__(self, robot_id, brain, body):
        self.check_compatibility(brain, body)
        self.id = robot_id
        self.brain = brain
        self.body = body
        self.body.id = robot_id

    def step(self):
        observations_map = self.body.get_observations()
        actuators_map = self.brain.step(observations_map)
        self.body.set_actuators(actuators_map)
        return observations_map, actuators_map
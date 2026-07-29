'''
    the setup of the program. Initializing simulator and stuff like that
'''

# IMPORTS
import numpy as np
import cv2
import pybullet as p
import pybullet_data
import time
import xacro


import statusboard as sb
import body
import brain
import robot

# Simulation Parameters
arenaname = "arena"
robotname = "Waldek"
robotbody = body.Epuck2Body
robotbrain = brain.AmebaPDDL
startPos = [-0.375, 0.3, 0.05]
startOrient = p.getQuaternionFromEuler([0, 0, 0])


# helper functions ---
def create_urdf_from_xacro(modelname):
    xacro_path = f"models/{modelname}.xacro"
    urdf_path = f"models/urdfs/{modelname}.urdf"
    doc = xacro.process_file(xacro_path)
    with open(urdf_path, "w") as f:
        f.write(doc.toxml())

    return urdf_path
# --------------------


# Config physics and environment
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.resetDebugVisualizerCamera(1.0, 90, -120, [0,0,0])
p.setGravity(0, 0, -9.81)

# load the arena
arena_urdf = create_urdf_from_xacro(arenaname)
arenaId = p.loadURDF(arena_urdf, [0, 0, 0], useFixedBase=True)
p.setJointMotorControl2(arenaId, 0, p.VELOCITY_CONTROL, targetVelocity=0, force=0)
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

# load the robot model
itsBody = robotbody("Waldek")
robot_urdf = create_urdf_from_xacro(itsBody.model)
robotId = p.loadURDF(robot_urdf, startPos, startOrient)

# create the robot object
itsBrain = robotbrain()
waldek = robot.Robot(robotId, itsBrain, itsBody)

# create the status board
board = sb.StatusBoard(robotname, itsBody.observations_map, itsBody.actuators_map)

# helpers for the loop
step_counter = 0
camera_skip = 10 
green_px, blue_px = 0, 0

# the loop
print("Symulacja uruchomiona")
debug_text_id = -1
try:
    while p.isConnected():
       
        [observations, actions] = waldek.step()
        p.stepSimulation()

        if(step_counter%3 == 0):
            board.update(observations, actions)

        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        
        step_counter += 1
        time.sleep(1./300.)

except KeyboardInterrupt:
    pass
finally:
    p.disconnect()
    cv2.destroyAllWindows()

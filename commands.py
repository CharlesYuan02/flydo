import threading
import time
from djitellopy import tello


def fly(direction, flydo):
    '''direction is a list of four variables dictating Flydo's flight actions.'''
    flydo.send_rc_control(direction[0], direction[1], direction[2], direction[3])
    time.sleep(0.05)


def start_flying(event, direction, flydo):
    '''Depending on what button user clicks, Flydo flies a certain way.'''
    lr, fb, ud, yv = 0, 0, 0, 0

    if direction == "upward":
        ud = 50
    elif direction == "downward":
        ud = -50
    elif direction == "forward":
        fb = 50
    elif direction == "backward":
        fb = -50
    elif direction == "yaw_left":
        yv = -50
    elif direction == "yaw_right":
        yv = 50
    elif direction == "left":
        lr = -50
    elif direction == "right":
        lr = 50
    elif direction == "flip_left":
        threading.Thread(target=lambda: flydo.flip("l")).start()
    elif direction == "flip_right":
        threading.Thread(target=lambda: flydo.flip("r")).start()
    
    if [lr, fb, ud, yv] != [0, 0, 0, 0]:
        threading.Thread(target=lambda: fly([lr, fb, ud, yv], flydo)).start()


def stop_flying(event, flydo):
    '''When user releases button, Flydo stops last flying action.'''
    flydo.send_rc_control(0, 0, 0, 0)
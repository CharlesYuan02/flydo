import cv2
import threading
import time
from djitellopy import tello
from tkinter import *
from PIL import Image, ImageTk # You have to import this last or else Image.open throws an error
from commands import start_flying, stop_flying


def takeoff_land(flydo):
    '''Flydo takes off if not flying, lands if flying.'''
    global flying
    if flying:
        # threading.Thread(target=lambda: dummy_tello_fn()).start()
        threading.Thread(target=lambda: flydo.land()).start()
        flying = False
    else:
        # threading.Thread(target=lambda: dummy_tello_fn()).start()
        threading.Thread(target=lambda: flydo.takeoff()).start()
        flying = True
    

def current_battery(flydo):
    '''Gets Flydo's current battery level every 5 seconds.
    Battery level displayed is altered if the battery decreases.'''
    global flydo_battery
    while True:
        flydo_battery = flydo.get_battery()
        time.sleep(5)


def run_app(HEIGHT=800, WIDTH=800):
    root = Tk()
    
    flydo = tello.Tello()
    flydo.connect()
    flydo.streamon()

    global flying
    flying = False # To toggle between takeoff and landing for button

    global flydo_battery
    bat_width = 200
    bat_height = 50
    flydo_battery = flydo.get_battery()
    threading.Thread(target=lambda: current_battery(flydo)).start()

    canvas = Canvas(root, height=HEIGHT, width=WIDTH)
    
    # For background image
    bg_dir = "C:\\Users\\charl\\Desktop\\flydo\\Tacit.jpg"
    img = Image.open(bg_dir).resize((WIDTH, HEIGHT))
    bg_label = Label(root)
    bg_label.img = ImageTk.PhotoImage(img)
    bg_label["image"] = bg_label.img
    bg_label.place(x=0, y=0, relwidth=1, relheight=1)

    # Takeoff/Land button
    takeoff_button = Button(root, text="Takeoff/Land", font=("Verdana", 18), bg="#95dff3", command=lambda: takeoff_land(flydo))
    tl_width = 200
    tl_height = 100
    takeoff_button.config(width=tl_width, height=tl_height)
    takeoff_button.place(relx=0.5-(bat_width/WIDTH)/2, rely=0.625, relwidth=tl_width/WIDTH, height=80)

    # Fly upwards button
    upward_button = Button(root, text=" /\ \n||", font=("Verdana", 18), bg="#95dff3")
    upward_button.place(x=100, rely=0.65, width=80, height=80)
    upward_button.bind("<ButtonPress-1>", lambda event: start_flying(event, "upward", flydo))
    upward_button.bind("<ButtonRelease-1>", lambda event: stop_flying(event, flydo))

    # Fly downwards button
    downward_button = Button(root, text=" || \n\/", font=("Verdana", 18), bg="#95dff3")
    downward_button.place(x=100, rely=0.85, width=80, height=80)
    downward_button.bind("<ButtonPress-1>", lambda event: start_flying(event, "downward", flydo))
    downward_button.bind("<ButtonRelease-1>", lambda event: stop_flying(event, flydo))

    # Fly forwards button
    forward_button = Button(root, text=" /\ \n||", font=("Verdana", 18), bg="#95dff3")
    forward_button.place(x=WIDTH-180, rely=0.65, width=80, height=80)
    forward_button.bind("<ButtonPress-1>", lambda event: start_flying(event, "forward", flydo))
    forward_button.bind("<ButtonRelease-1>", lambda event: stop_flying(event, flydo))

    # Fly backwards button
    backward_button = Button(root, text=" || \n\/", font=("Verdana", 18), bg="#95dff3")
    backward_button.place(x=WIDTH-180, rely=0.85, width=80, height=80)
    backward_button.bind("<ButtonPress-1>", lambda event: start_flying(event, "backward", flydo))
    backward_button.bind("<ButtonRelease-1>", lambda event: stop_flying(event, flydo))

    # Yaw left button
    yawleft_button = Button(root, text="<=", font=("Verdana", 18), bg="#95dff3")
    yawleft_button.place(x=20, rely=(0.85+0.65)/2, width=80, height=80)
    yawleft_button.bind("<ButtonPress-1>", lambda event: start_flying(event, "yaw_left", flydo))
    yawleft_button.bind("<ButtonRelease-1>", lambda event: stop_flying(event, flydo))

    # Yaw right button
    yawright_button = Button(root, text="=>", font=("Verdana", 18), bg="#95dff3")
    yawright_button.place(x=180, rely=(0.85+0.65)/2, width=80, height=80)
    yawright_button.bind("<ButtonPress-1>", lambda event: start_flying(event, "yaw_right", flydo))
    yawright_button.bind("<ButtonRelease-1>", lambda event: stop_flying(event, flydo))

    # Fly left button
    flyleft_button = Button(root, text="<=", font=("Verdana", 18), bg="#95dff3")
    flyleft_button.place(x=WIDTH-260, rely=(0.85+0.65)/2, width=80, height=80)
    flyleft_button.bind("<ButtonPress-1>", lambda event: start_flying(event, "left", flydo))
    flyleft_button.bind("<ButtonRelease-1>", lambda event: stop_flying(event, flydo))

    # Fly right button
    flyright_button = Button(root, text="=>", font=("Verdana", 18), bg="#95dff3")
    flyright_button.place(x=WIDTH-100, rely=(0.85+0.65)/2, width=80, height=80)
    flyright_button.bind("<ButtonPress-1>", lambda event: start_flying(event, "right", flydo))
    flyright_button.bind("<ButtonRelease-1>", lambda event: stop_flying(event, flydo))

    # Flip left button
    flipleft_button = Button(root, text="<<", font=("Verdana", 18), bg="#95dff3")
    flipleft_button.place(x=100, rely=(0.85+0.65)/2, width=80, height=80)
    flipleft_button.bind("<ButtonPress-1>", lambda event: start_flying(event, "flip_left", flydo))
    flipleft_button.bind("<ButtonRelease-1>", lambda event: stop_flying(event, flydo))

    # Flip right button
    flipright_button = Button(root, text=">>", font=("Verdana", 18), bg="#95dff3")
    flipright_button.place(x=WIDTH-180, rely=(0.85+0.65)/2, width=80, height=80)
    flipright_button.bind("<ButtonPress-1>", lambda event: start_flying(event, "flip_right", flydo))
    flipright_button.bind("<ButtonRelease-1>", lambda event: stop_flying(event, flydo))

    '''
    The following three buttons are work-in-progress.
    '''

    # Take picture button
    picture_button = Button(root, text="•", font=("Verdana", 18), bg="#95dff3")
    picture_button.place(x=280, rely=(0.85+0.65)/2, width=80, height=80)

    # Take video button
    video_button = Button(root, fg="red", text="•", font=("Verdana", 18), bg="#95dff3")
    video_button.place(x=440, rely=(0.85+0.65)/2, width=80, height=80)

    # Track person button
    tracking_button = Button(root, text="⌐╦╦═─", font=("Verdana", 16), bg="#95dff3")
    tracking_button.place(x=360, rely=(0.85+0.65)/2, width=80, height=80)

    def display_battery():
        '''Displays and updates current battery level.'''
        battery = Label(text=f"Battery: {flydo_battery}%", font=("Verdana", 18), bg="#95dff3")
        battery.config(width=bat_width, height=bat_height)
        battery.place(relx=0.5-(bat_width/WIDTH)/2, rely=0.875, relwidth=bat_width/WIDTH, height=80)
        root.after(5, display_battery)
    display_battery()

    cap_label = Label(root)
    cap_label.pack()
    
    def video_stream():
        h = 480
        w = 720
        frame = flydo.get_frame_read().frame
        frame = cv2.resize(frame, (w, h))
        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        img = Image.fromarray(cv2image)
        imgtk = ImageTk.PhotoImage(image=img)
        cap_label.place(x=WIDTH/2 - w/2, y=0)
        cap_label.imgtk = imgtk
        cap_label.configure(image=imgtk)
        cap_label.after(5, video_stream)

    video_stream()
    canvas.pack()
    root.mainloop()


if __name__ == "__main__":
    HEIGHT = 800
    WIDTH = 800

    run_app(HEIGHT, WIDTH)

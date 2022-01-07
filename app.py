import cv2
import datetime
import numpy as np
import os
import threading
import time
import torch
from djitellopy import tello
from moviepy.editor import *
from tkinter import *
from PIL import Image, ImageTk # You have to import this last or else Image.open throws an error
from commands import start_flying, stop_flying
from model.train import get_transform


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


def take_screenshot(img):
    '''Takes whatever Flydo current sees and saves the original image (before GUI resizing).'''
    global screenshot_taken
    if not os.path.exists("screenshots"):
        os.mkdir("screenshots")
    pic_name = str(datetime.datetime.now())
    pic_name = pic_name.replace(":", "-").replace(" ", "-").replace(".", "-")
    threading.Thread(target=lambda: cv2.imwrite(f"screenshots/{pic_name}.png", img)).start()
    screenshot_taken = True


def take_video():
    '''Click record button once to start recording, click again to stop and save video.'''
    global recording
    global current_img
    global vid_name
    if not os.path.exists("recordings"):
        os.mkdir("recordings")
    if not os.path.exists("unprocessed_recordings"):
        os.mkdir("unprocessed_recordings")

    if recording:
        recording = False

        # Post-processing because Flydo saved video is 5X slower for some reason.
        clip = VideoFileClip(f"unprocessed_recordings/{vid_name}.avi")
        final = clip.fx(vfx.speedx, 5)
        threading.Thread(target=lambda: final.write_videofile(f"recordings/{vid_name}.mp4")).start() # Speed up video 5X, save as mp4

    else:
        # First remove old unprocessed videos to save space
        for vid in os.listdir("unprocessed_recordings"):
            os.remove(f"unprocessed_recordings/{vid}")
        recording = True
        threading.Thread(target=lambda: take_video_helper()).start()


def take_video_helper():
    '''This is just to make threading easier, cuz creating the video uses a while loop.'''
    global recording
    global vid_name
    vid_name = str(datetime.datetime.now())
    vid_name = vid_name.replace(":", "-").replace(" ", "-").replace(".", "-")
    videowriter = cv2.VideoWriter(f"unprocessed_recordings/{vid_name}.avi", cv2.VideoWriter_fourcc('M','J','P','G'), 30, (current_img.shape[1], current_img.shape[0]))

    i = 0
    while recording:
        frame = current_img
        videowriter.write(frame)    


def track_person(model_path):
    '''Uses trained Faster R-CNN model to create bounding box,
    positions center of Flydo's POV towards center of bbox.'''
    global tracking

    if tracking:
        tracking = False
        print("Stopped tracking")
    else:
        tracking = True
        threading.Thread(target=lambda: tracking_helper(model_path)).start()
        print("Now tracking")
    

def tracking_helper(model_path):
    '''Just a helper function for threading purposes.'''
    global current_img
    global tracking_img
    global tracking

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = torch.load(model_path)
    model.eval() # Set to evaluation mode
    transform = get_transform()
    
    while tracking:
        with torch.no_grad():
            img = Image.fromarray(cv2.cvtColor(current_img, cv2.COLOR_BGR2RGB)) # Convert cv2 img to PIL
            img = transform(img).to(device)
            img = img.unsqueeze(0) # Add an extra first dimension, because the model predicts in batches
            prediction = model(img)
            try:
                max_idx = np.argmax(prediction[0]["scores"].cpu().numpy())
                confidence = prediction[0]["scores"][max_idx].cpu().numpy()
                bbox_coords = prediction[0]["boxes"].cpu().numpy()[max_idx]
                bbox_coords = [int(x) for x in bbox_coords]
                image = current_img
                image = cv2.rectangle(image, (bbox_coords[0], bbox_coords[1]), (bbox_coords[2], bbox_coords[3]), (0, 0, 255), 2)
                image = cv2.putText(image, f"{int(confidence*100)}%", (bbox_coords[0], bbox_coords[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            except ValueError: # If there are no boxes detected
                pass
            image = cv2.putText(image, "Now Tracking", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            tracking_img = image # This will be the new overlay


def run_app(HEIGHT=800, WIDTH=800):
    root = Tk()
    canvas = Canvas(root, height=HEIGHT, width=WIDTH)
    
    # To connect to the drone and read in image feed
    flydo = tello.Tello()
    flydo.connect()
    flydo.streamon()

    # For background image
    bg_dir = "C:\\Users\\charl\\Desktop\\flydo\\Tacit.jpg"
    img = Image.open(bg_dir).resize((WIDTH, HEIGHT))
    bg_label = Label(root)
    bg_label.img = ImageTk.PhotoImage(img)
    bg_label["image"] = bg_label.img
    bg_label.place(x=0, y=0, relwidth=1, relheight=1)

    # To toggle between takeoff and landing for button
    global flying
    flying = False

    # For updating Flydo battery reading
    global flydo_battery
    bat_width = 200
    bat_height = 50
    flydo_battery = flydo.get_battery()
    threading.Thread(target=lambda: current_battery(flydo)).start()

    # For taking screenshots
    global screenshot_taken
    screenshot_taken = False

    # For recording videos
    global recording
    recording = False

    # For tracking function
    global tracking
    global tracking_img
    tracking = False
    tracking_img = None
    MODEL_PATH = "model/trained_detector.pth"

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

    # Take picture button
    picture_button = Button(root, text="•", font=("Verdana", 18), bg="#95dff3", command=lambda: take_screenshot(current_img))
    picture_button.place(x=280, rely=(0.85+0.65)/2, width=80, height=80)

    # Take video button
    video_button = Button(root, fg="red", text="•", font=("Verdana", 18), bg="#95dff3", command=lambda: take_video())
    video_button.place(x=440, rely=(0.85+0.65)/2, width=80, height=80)

    # Track person button
    tracking_button = Button(root, text="⌐╦╦═─", font=("Verdana", 16), bg="#95dff3", command=lambda: track_person(MODEL_PATH))
    tracking_button.place(x=360, rely=(0.85+0.65)/2, width=80, height=80)

    def display_battery():
        '''Displays and updates current battery level.
        Battery level is replaced with "Screenshotted" briefly when a screenshot is taken successfully.'''
        global screenshot_taken
        if screenshot_taken:
            battery = Label(text=f"Screenshotted", font=("Verdana", 18), bg="#95dff3")
            battery.config(width=bat_width, height=bat_height)
            battery.place(relx=0.5-(bat_width/WIDTH)/2, rely=0.875, relwidth=bat_width/WIDTH, height=80)
            root.after(500, display_battery)
            screenshot_taken = False
        elif recording:
            battery = Label(text=f"Recording", font=("Verdana", 18), bg="#95dff3")
            battery.config(width=bat_width, height=bat_height)
            battery.place(relx=0.5-(bat_width/WIDTH)/2, rely=0.875, relwidth=bat_width/WIDTH, height=80)
            root.after(5, display_battery)
        else:
            battery = Label(text=f"Battery: {int(flydo_battery)}%", font=("Verdana", 18), bg="#95dff3")
            battery.config(width=bat_width, height=bat_height)
            battery.place(relx=0.5-(bat_width/WIDTH)/2, rely=0.875, relwidth=bat_width/WIDTH, height=80)
            root.after(5, display_battery)
    display_battery()

    # For video stream
    cap_label = Label(root)
    cap_label.pack()
    
    def video_stream():
        global current_img
        global tracking
        global tracking_img
        h = 480
        w = 720
        frame = flydo.get_frame_read().frame
        current_img = frame # For taking screenshots

        if tracking and tracking_img is not None:
            frame = cv2.resize(tracking_img, (w, h))
        else:
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

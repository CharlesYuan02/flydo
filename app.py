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


class GUI():
    def __init__(self, HEIGHT, WIDTH):
        self.HEIGHT = HEIGHT
        self.WIDTH = WIDTH
        
        self.root = Tk()
        self.canvas = Canvas(self.root, height=self.HEIGHT, width=self.WIDTH)

        # To connect to the drone and read in image feed
        self.flydo = tello.Tello()
        self.flydo.connect()
        self.flydo.streamon()

        # To toggle between takeoff and landing for button
        self.flying = False

        # For updating Flydo battery reading
        self.bat_width = 200
        self.bat_height = 50
        self.flydo_battery = self.flydo.get_battery()
        threading.Thread(target=lambda: self.current_battery()).start()

        # For taking screenshots
        self.screenshot_taken = False

        # For recording videos
        self.recording = False

        # For tracking function
        self.tracking = False
        self.tracking_img = None
        self.MODEL_PATH = "model/trained_detector.pth"
    
    def takeoff_land(self):
        '''Flydo takes off if not flying, lands if flying.'''
        if self.flying:
            # threading.Thread(target=lambda: dummy_tello_fn()).start()
            threading.Thread(target=lambda: self.flydo.land()).start()
            self.flying = False
        else:
            # threading.Thread(target=lambda: dummy_tello_fn()).start()
            threading.Thread(target=lambda: self.flydo.takeoff()).start()
            self.flying = True
    
    def current_battery(self):
        '''Gets Flydo's current battery level every 5 seconds.
        Battery level displayed is altered if the battery decreases.'''
        while True:
            self.flydo_battery = self.flydo.get_battery()
            time.sleep(5)

    def take_screenshot(self):
        '''Takes whatever Flydo current sees and saves the original image (before GUI resizing).'''
        if not os.path.exists("screenshots"):
            os.mkdir("screenshots")
        pic_name = str(datetime.datetime.now())
        pic_name = pic_name.replace(":", "-").replace(" ", "-").replace(".", "-")
        threading.Thread(target=lambda: cv2.imwrite(f"screenshots/{pic_name}.png", self.current_img)).start()
        self.screenshot_taken = True
    
    def take_video(self):
        '''Click record button once to start recording, click again to stop and save video.'''
        if not os.path.exists("recordings"):
            os.mkdir("recordings")
        if not os.path.exists("unprocessed_recordings"):
            os.mkdir("unprocessed_recordings")

        if self.recording:
            self.recording = False

            # Post-processing because Flydo saved video is 5X slower for some reason.
            clip = VideoFileClip(f"unprocessed_recordings/{self.vid_name}.avi")
            final = clip.fx(vfx.speedx, 5)
            threading.Thread(target=lambda: final.write_videofile(f"recordings/{self.vid_name}.mp4")).start() # Speed up video 5X, save as mp4

        else:
            # First remove old unprocessed videos to save space
            for vid in os.listdir("unprocessed_recordings"):
                os.remove(f"unprocessed_recordings/{vid}")
            self.recording = True
            threading.Thread(target=lambda: self.take_video_helper()).start()
    
    def take_video_helper(self):
        '''This is just to make threading easier, cuz creating the video uses a while loop.'''
        self.vid_name = str(datetime.datetime.now())
        self.vid_name = self.vid_name.replace(":", "-").replace(" ", "-").replace(".", "-")
        videowriter = cv2.VideoWriter(f"unprocessed_recordings/{self.vid_name}.avi", cv2.VideoWriter_fourcc('M','J','P','G'), 30, (self.current_img.shape[1], self.current_img.shape[0]))

        i = 0
        while self.recording:
            frame = self.current_img
            videowriter.write(frame)    


    def track_person(self):
        '''Uses trained Faster R-CNN model to create bounding box,
        positions center of Flydo's POV towards center of bbox.'''
        if self.tracking:
            self.tracking = False
            print("Stopped tracking")
        else:
            self.tracking = True
            threading.Thread(target=lambda: self.tracking_helper()).start()
            print("Now tracking")
    
    def tracking_helper(self):
        '''Just a helper function for threading purposes.'''
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        model = torch.load(self.MODEL_PATH)
        model.eval() # Set to evaluation mode
        transform = get_transform()
        
        while self.tracking:
            with torch.no_grad():
                img = Image.fromarray(cv2.cvtColor(self.current_img, cv2.COLOR_BGR2RGB)) # Convert cv2 img to PIL
                img = transform(img).to(device)
                img = img.unsqueeze(0) # Add an extra first dimension, because the model predicts in batches
                prediction = model(img)
                try:
                    max_idx = np.argmax(prediction[0]["scores"].cpu().numpy())
                    confidence = prediction[0]["scores"][max_idx].cpu().numpy()
                    bbox_coords = prediction[0]["boxes"].cpu().numpy()[max_idx]
                    bbox_coords = [int(x) for x in bbox_coords]
                    image = self.current_img
                    image = cv2.rectangle(image, (bbox_coords[0], bbox_coords[1]), (bbox_coords[2], bbox_coords[3]), (0, 0, 255), 2)
                    image = cv2.putText(image, f"{int(confidence*100)}%", (bbox_coords[0], bbox_coords[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                except ValueError: # If there are no boxes detected
                    pass
                image = cv2.putText(image, "Now Tracking", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                self.tracking_img = image # This will be the new overlay

    def run_app(self):
        # For background image
        bg_dir = "C:\\Users\\charl\\Desktop\\flydo\\docs\\Tacit.jpg"
        img = Image.open(bg_dir).resize((self.WIDTH, self.HEIGHT))
        bg_label = Label(self.root)
        bg_label.img = ImageTk.PhotoImage(img)
        bg_label["image"] = bg_label.img
        bg_label.place(x=0, y=0, relwidth=1, relheight=1)

        # Takeoff/Land button
        takeoff_button = Button(self.root, text="Takeoff/Land", font=("Verdana", 18), bg="#95dff3", command=lambda: self.takeoff_land())
        tl_width = 200
        tl_height = 100
        takeoff_button.config(width=tl_width, height=tl_height)
        takeoff_button.place(relx=0.5-(self.bat_width/self.WIDTH)/2, rely=0.625, relwidth=tl_width/self.WIDTH, height=80)

        # Fly upwards button
        upward_button = Button(self.root, text=" /\ \n||", font=("Verdana", 18), bg="#95dff3")
        upward_button.place(x=100, rely=0.65, width=80, height=80)
        upward_button.bind("<ButtonPress-1>", lambda event: start_flying(event, "upward", self.flydo))
        upward_button.bind("<ButtonRelease-1>", lambda event: stop_flying(event, self.flydo))

        # Fly downwards button
        downward_button = Button(self.root, text=" || \n\/", font=("Verdana", 18), bg="#95dff3")
        downward_button.place(x=100, rely=0.85, width=80, height=80)
        downward_button.bind("<ButtonPress-1>", lambda event: start_flying(event, "downward", self.flydo))
        downward_button.bind("<ButtonRelease-1>", lambda event: stop_flying(event, self.flydo))

        # Fly forwards button
        forward_button = Button(self.root, text=" /\ \n||", font=("Verdana", 18), bg="#95dff3")
        forward_button.place(x=self.WIDTH-180, rely=0.65, width=80, height=80)
        forward_button.bind("<ButtonPress-1>", lambda event: start_flying(event, "forward", self.flydo))
        forward_button.bind("<ButtonRelease-1>", lambda event: stop_flying(event, self.flydo))

        # Fly backwards button
        backward_button = Button(self.root, text=" || \n\/", font=("Verdana", 18), bg="#95dff3")
        backward_button.place(x=self.WIDTH-180, rely=0.85, width=80, height=80)
        backward_button.bind("<ButtonPress-1>", lambda event: start_flying(event, "backward", self.flydo))
        backward_button.bind("<ButtonRelease-1>", lambda event: stop_flying(event, self.flydo))

        # Yaw left button
        yawleft_button = Button(self.root, text="<=", font=("Verdana", 18), bg="#95dff3")
        yawleft_button.place(x=20, rely=(0.85+0.65)/2, width=80, height=80)
        yawleft_button.bind("<ButtonPress-1>", lambda event: start_flying(event, "yaw_left", self.flydo))
        yawleft_button.bind("<ButtonRelease-1>", lambda event: stop_flying(event, self.flydo))

        # Yaw right button
        yawright_button = Button(self.root, text="=>", font=("Verdana", 18), bg="#95dff3")
        yawright_button.place(x=180, rely=(0.85+0.65)/2, width=80, height=80)
        yawright_button.bind("<ButtonPress-1>", lambda event: start_flying(event, "yaw_right", self.flydo))
        yawright_button.bind("<ButtonRelease-1>", lambda event: stop_flying(event, self.flydo))

        # Fly left button
        flyleft_button = Button(self.root, text="<=", font=("Verdana", 18), bg="#95dff3")
        flyleft_button.place(x=self.WIDTH-260, rely=(0.85+0.65)/2, width=80, height=80)
        flyleft_button.bind("<ButtonPress-1>", lambda event: start_flying(event, "left", self.flydo))
        flyleft_button.bind("<ButtonRelease-1>", lambda event: stop_flying(event, self.flydo))

        # Fly right button
        flyright_button = Button(self.root, text="=>", font=("Verdana", 18), bg="#95dff3")
        flyright_button.place(x=self.WIDTH-100, rely=(0.85+0.65)/2, width=80, height=80)
        flyright_button.bind("<ButtonPress-1>", lambda event: start_flying(event, "right", self.flydo))
        flyright_button.bind("<ButtonRelease-1>", lambda event: stop_flying(event, self.flydo))

        # Flip left button
        flipleft_button = Button(self.root, text="<<", font=("Verdana", 18), bg="#95dff3")
        flipleft_button.place(x=100, rely=(0.85+0.65)/2, width=80, height=80)
        flipleft_button.bind("<ButtonPress-1>", lambda event: start_flying(event, "flip_left", self.flydo))
        flipleft_button.bind("<ButtonRelease-1>", lambda event: stop_flying(event, self.flydo))

        # Flip right button
        flipright_button = Button(self.root, text=">>", font=("Verdana", 18), bg="#95dff3")
        flipright_button.place(x=self.WIDTH-180, rely=(0.85+0.65)/2, width=80, height=80)
        flipright_button.bind("<ButtonPress-1>", lambda event: start_flying(event, "flip_right", self.flydo))
        flipright_button.bind("<ButtonRelease-1>", lambda event: stop_flying(event, self.flydo))

        # Take picture button
        picture_button = Button(self.root, text="•", font=("Verdana", 18), bg="#95dff3", command=lambda: self.take_screenshot())
        picture_button.place(x=280, rely=(0.85+0.65)/2, width=80, height=80)

        # Take video button
        video_button = Button(self.root, fg="red", text="•", font=("Verdana", 18), bg="#95dff3", command=lambda: self.take_video())
        video_button.place(x=440, rely=(0.85+0.65)/2, width=80, height=80)

        # Track person button
        tracking_button = Button(self.root, text="⌐╦╦═─", font=("Verdana", 16), bg="#95dff3", command=lambda: self.track_person())
        tracking_button.place(x=360, rely=(0.85+0.65)/2, width=80, height=80)

        # For video stream
        self.cap_label = Label(self.root)
        self.cap_label.pack()

        self.display_battery()
        self.video_stream()
        self.canvas.pack()
        self.root.mainloop()

    def display_battery(self):
        '''Displays and updates current battery level.
        Battery level is replaced with "Screenshotted" briefly when a screenshot is taken successfully.'''
        if self.screenshot_taken:
            battery = Label(text=f"Screenshotted", font=("Verdana", 18), bg="#95dff3")
            battery.config(width=self.bat_width, height=self.bat_height)
            battery.place(relx=0.5-(self.bat_width/self.WIDTH)/2, rely=0.875, relwidth=self.bat_width/self.WIDTH, height=80)
            self.root.after(500, self.display_battery)
            self.screenshot_taken = False
        elif self.recording:
            battery = Label(text=f"Recording", font=("Verdana", 18), bg="#95dff3")
            battery.config(width=self.bat_width, height=self.bat_height)
            battery.place(relx=0.5-(self.bat_width/self.WIDTH)/2, rely=0.875, relwidth=self.bat_width/self.WIDTH, height=80)
            self.root.after(5, self.display_battery)
        else:
            battery = Label(text=f"Battery: {int(self.flydo_battery)}%", font=("Verdana", 18), bg="#95dff3")
            battery.config(width=self.bat_width, height=self.bat_height)
            battery.place(relx=0.5-(self.bat_width/self.WIDTH)/2, rely=0.875, relwidth=self.bat_width/self.WIDTH, height=80)
            self.root.after(5, self.display_battery)

    def video_stream(self):
        h = 480
        w = 720
        frame = self.flydo.get_frame_read().frame
        self.current_img = frame # For taking screenshots

        if self.tracking and self.tracking_img is not None:
            frame = cv2.resize(self.tracking_img, (w, h))
        else:
            frame = cv2.resize(frame, (w, h))
        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        img = Image.fromarray(cv2image)
        imgtk = ImageTk.PhotoImage(image=img)
        self.cap_label.place(x=self.WIDTH/2 - w/2, y=0)
        self.cap_label.imgtk = imgtk
        self.cap_label.configure(image=imgtk)
        self.cap_label.after(5, self.video_stream)


if __name__ == "__main__":
    FlydoGUI = GUI(800, 800)
    FlydoGUI.run_app()

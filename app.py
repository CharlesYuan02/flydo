import cv2
import threading
from djitellopy import tello
from tkinter import *
from PIL import Image, ImageTk # You have to import this last or else Image.open throws an error

def takeoff_land(flydo):
    '''Flydo takes off if not flying, lands if flying.'''
    global flying
    if flying:
        threading.Thread(target=lambda: flydo.land()).start()
        flying = False
    else:
        threading.Thread(target=lambda: flydo.takeoff()).start()
        flying = True
    

def run_app(HEIGHT=800, WIDTH=800):
    root = Tk()
    
    flydo = tello.Tello()
    flydo.connect()
    flydo.streamon()

    global flying
    flying = False # To toggle between takeoff and landing for button

    canvas = Canvas(root, height=HEIGHT, width=WIDTH)
    
    # For background image
    bg_dir = "C:\\Users\\charl\\Desktop\\flydo\\Tacit.jpg"
    img = Image.open(bg_dir).resize((WIDTH, HEIGHT))
    bg_label = Label(root)
    bg_label.img = ImageTk.PhotoImage(img)
    bg_label["image"] = bg_label.img
    bg_label.place(x=0, y=0, relwidth=1, relheight=1)

    # Display current battery
    battery = Label(text=f"Battery: {int(flydo.get_battery())}%", font=("Verdana", 18), bg="#95dff3")
    bat_width = 200
    bat_height = 50
    battery.config(width=bat_width, height=bat_height)
    battery.place(x=(WIDTH - bat_width - 0.1*HEIGHT + bat_height), rely=0.9, relwidth=bat_width/WIDTH, relheight=bat_height/HEIGHT)

    # Takeoff/Land button
    forward_button = Button(root, text="Takeoff/Land", font=("Verdana", 18), bg="#95dff3", command=lambda: takeoff_land(flydo))
    fb_width = 200
    fb_height = 100
    forward_button.config(width=fb_width, height=fb_height)
    forward_button.place(x=(WIDTH/2 - fb_width/2), rely=0.61, relwidth=fb_width/WIDTH, relheight=fb_height/HEIGHT)

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

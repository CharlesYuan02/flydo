# flydo
<img src="https://github.com/Chubbyman2/flydo/blob/main/flydo.PNG">

## Introduction
I have no idea why, but back when 2021 first started, I was listening to <a href="https://www.youtube.com/watch?v=8ajBxCch0No">Speed of Light by DJ Okawari</a> (you know, that Arknights song). The video was basically just a looping gif of Texas flying a drone with Exusiai, and for some reason I ended up really wanting a drone. So I bought one from DJI that could be programmed with Python, the Tello EDU. I've been meaning to do this project for a while now, so here it is. A GUI made using Tkinter for my DJI Tello EDU, named Flydo (after <a href="https://86-eighty-six.fandom.com/wiki/Fido">Fido</a> from 86).

## Prerequisites
```
djitellopy==2.4.0
python-opencv==4.5.5.62
```

## Built With
### djitellopy (Tello EDU SDK)
djitellopy was implemented based on the official DJI Tello/Tello EDU SDK, and allows for easy implementation of all tello commands, simple retrieval of the drone's video stream, and parsing/receiving of state packets. All drone commands were implemented using this SDK.

### Tkinter
Tkinter was used to create the GUI for controlling the drone. Different drone commands were implemented with multithreading so as to prevent the visual display from freezing while the drone is moving (this took me a while to figure out how to do). The result is shown below.

## Tkinter Graphical User Interface (GUI)
<img src="https://github.com/Chubbyman2/flydo/blob/main/gui.PNG">

## Future Plans
1. Implement functions for taking pictures and videos.
2. Use an object detection algorithm (YOLOv5, etc) to allow the drone to track humans.
3. Figure out how to decrease latency between drone feed and visual display. 

## License
This project is licensed under the MIT License - see the <a href="https://github.com/Chubbyman2/flydo/blob/main/LICENSE">LICENSE</a> file for details.

# flydo
<img src="https://github.com/Chubbyman2/flydo/blob/main/docs/flydo.PNG">

## Introduction
I have no idea why, but back when 2021 first started, I was listening to <a href="https://www.youtube.com/watch?v=8ajBxCch0No">Speed of Light by DJ Okawari</a> (you know, that Arknights song). The video was basically just a looping gif of Texas flying a drone with Exusiai, and for some reason I ended up really wanting a drone. So I bought one from DJI that could be programmed with Python, the Tello EDU. I've been meaning to do this project for a while now, so here it is. A GUI made using Tkinter for my DJI Tello EDU, named Flydo (after <a href="https://86-eighty-six.fandom.com/wiki/Fido">Fido</a> from 86).

## Getting Started
To get started, you first need a trained Pytorch model (since I couldn't upload the large saved .pth file).
```
model/train.py
```
Then you can run the actual GUI: 
```
app.py
```

### Prerequisites
```
-f https://download.pytorch.org/whl/torch_stable.html
torch==1.10.1+cu113 
torchvision==0.11.2+cu113 
torchaudio===0.10.1+cu113
djitellopy==2.4.0
pycocotools-windows==2.0.0.2
pygame==2.1.2
python-opencv==4.5.5.62
```

## Built With
### djitellopy (Tello EDU SDK)
djitellopy was implemented based on the official DJI Tello/Tello EDU SDK, and allows for easy implementation of all tello commands, simple retrieval of the drone's video stream, and parsing/receiving of state packets. All drone commands were implemented using this SDK.

### Tkinter
Tkinter was used to create the GUI for controlling the drone. Different drone commands were implemented with multithreading so as to prevent the visual display from freezing while the drone is moving (this took me a while to figure out how to do). The result is shown below.

### Pytorch
Pytorch's pretrained Faster R-CNN model was used for the head-tracking feature. The model was fine-tuned using transfer learning on a custom dataset. It actually works a lot better than I thought it would, especially given the limited amount of training data and epochs! 

### Coco-Annotator
<a href="https://github.com/jsbroks/coco-annotator">Coco-Annotator<a> is an image-labelling tool created by felixdollack that I also use for my research. It outputs data in Coco format for object detection and instance segmentation, and I used it to label my <a href="https://github.com/Chubbyman2/flydo/tree/main/model/labelled_data">custom dataset</a> that I collated using Flydo's own camera. It's basically just a bunch of pictures of me in my room.

## Tkinter Graphical User Interface (GUI)
<img src="https://github.com/Chubbyman2/flydo/blob/main/docs/gui.PNG">

## Future Plans
### Update 1 (Jan 5, 2022):
Flydo can now take screenshots and videos! 

### Update 2 (Jan 6, 2022):
Flydo's tracking function now generates and overlays bounding boxes!
  
### Update 3 (Jan 7, 2022):
app.py's associated functions have been reformatted into class methods. All global variables have been removed and replaced as class variables.
  
### Update 4 (Mar 3, 2022):
Apparently the issue with the video feed latency has to do with the djitellopy library itself, and there isn't a way to avoid this unless you switch to <a href="https://github.com/hanyazou/TelloPy">TelloPy</a>. I'm currently in the middle of midterms, but I'll make sure to do this soon.

### To Do:
1. Create a function to move Flydo based on output of object detector.
2. Improve accuracy of tracker (i.e. the object detection model).
3. Figure out how to decrease latency between drone feed and visual display. 

## Acknowledgements
Thanks to Takashi Nakamura, PhD, for writing this <a href="https://medium.com/fullstackai/how-to-train-an-object-detector-with-your-own-coco-dataset-in-pytorch-319e7090da5">article</a> showing me the basics on how to train Faster R-CNN with Pytorch.

## License
This project is licensed under the MIT License - see the <a href="https://github.com/Chubbyman2/flydo/blob/main/LICENSE">LICENSE</a> file for details.

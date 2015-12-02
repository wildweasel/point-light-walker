# point-light-walker
Generate point light walker data

## Prequistes
* Python 3
* Tkinter
* PIL/Pillow
* ImageTk
* Numpy
* OpenCV 3 w/ Python 3 bindings

## Status
Opens video file.  Plays & Replays video.  Allows speed control.

## Issues
At full speed, flashes back to inital video frame.  Doesn't affect order of frames provided to processor

## Controls
* "Load Video Clip" selects and runs video clip
* "Run" re-runs loaded video clip
* "Speed" spinbox allows selection of inter-frame delay (in seconds)

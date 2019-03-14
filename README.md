# Fit Dlib 68 to images or video

Simple fitting of dlib 68 landmarks - Fits only to the first face detected in a video frame.
We actually detect the landmarks twice - once in the flipped image - and return the
mean of the two fittings.
There is a utility to extract frames to images too.

## Requirements

* dlib
* click
* imageio
* numpy
* pkg_resources

## Install

From the source directory:

    pip install -e .

There are entry points created for the command line.
It is necessary to manually install the dlib model `shape_predictor_68_face_landmarks.dat` .

## Examples

To detect landmarks.

    dlib-fit --verbose 3 --start 0 --end 30 \
        "/Volumes/data1/AJEMO-R/scene-01/001_0002.mp4" \
        "./scene-01/001_0002/"

To extract and save video frames.

    vid2png --verbose 3 --start 0 --end 30 \
        "/Volumes/data1/AJEMO-R/scene-01/001_0002.mp4" \
        "./scene-01/001_0002/"

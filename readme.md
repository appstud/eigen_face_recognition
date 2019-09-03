## part 1. Introduction

Face recognition based on eigen face. 
Having a database of images of people, this code detects the faces as well as landmarks on the face using the dlib library.

It then preprocess the images and aligns them using a 2D similarity transform and apply PCA and K nearest neighbor for classification.

It draws the first 10 eigen faces and it then applies the detection+alignment+recognition code on the video stream of the webcam.
 
You are supposed  to install some dependencies before getting out hands with these codes:

OpenCV3.0.1
dlib
sklearn
imutils
scipy

## part 2. Quick start

1. Build a database of images and organize it under a folder called data.
   make a subfolder inside the data folder and name it by the name of a person
   put all the images of this person inside

2. clone this code in the root folder next to the data folder

3. Run the code face_recognition_eigenFaces

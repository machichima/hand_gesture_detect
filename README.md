# hand_gesture_detect
## Description
Use python tensorflow Keras to build CNN and LSTM model, use them to predict static and dynamic hand gesture.

## Data preparation
use file change_img_to_YCrCb.py to change pictures into black-and-white pictures, As it is more simple to identify through black-and-white picture.

## using CNN
Using CNN to identify static hand gesture
- neural network structure:
<img src="https://user-images.githubusercontent.com/60069744/145400090-aeb8d99d-c6ed-435c-a31f-ac1d10b3ff9e.png" width="200">

### Result
Get test accuracy up to 0.75.
- testing result:
<img src="https://user-images.githubusercontent.com/60069744/145404120-bc24b86a-7224-4f26-9f13-af8800d20c06.png" height="50">

- prediction compare to real label:
<img src="https://user-images.githubusercontent.com/60069744/145404316-a7725495-686c-41ee-85b2-78c59ecdada8.png" width="200">


Because of using black-and-white picture, the prediction quality isn't good for some similar pictures (see example below):

<img src="https://user-images.githubusercontent.com/60069744/145401534-40858d46-4639-4232-b371-d35d3084b5c4.png" width="200">

## using LSTM
Using LSTM to identify dynamic hand gesture.
- neural network structure:
<img src="https://user-images.githubusercontent.com/60069744/145401869-4bcbe509-6e53-40d6-95c7-28204ac51344.png" width="250">

### make a series of data
Put 10 pictures in a group and label them.
Add "swapping left" and "swapping right" gesture (see pictures below), labeled as 11 and 12.
- swapping left:
<img src="https://user-images.githubusercontent.com/60069744/145402700-630af9e3-68f2-400c-a087-309c4acf45c2.png" height="50">

- swapping right:
<img src="https://user-images.githubusercontent.com/60069744/145402790-273cea56-35bd-4194-8f23-642a141b5463.png" height="50">

### Result
Get test accuracy up to 0.6538.
- testing result:
<img src="https://user-images.githubusercontent.com/60069744/145403764-747f9bfa-344b-4093-b6f6-fa6768d05204.png" height="50">

- prediction compare to real label:
<img src="https://user-images.githubusercontent.com/60069744/145403153-5a077b71-77d8-4a66-8010-85207ccc90b1.png" height="250">

LSTM model can predict dynamic gesture well (label 11 and 12), but would made more error on identifying static gesture.

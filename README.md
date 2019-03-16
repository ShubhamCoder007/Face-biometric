# Face-biometric
Detects the face and identifies using CNN for the biometrics.

This is a gui based program which has got some options to start with.
At first we need datas to train the CNN in order to identify and predict correctly.
So, a create profile section is there where the directory for the student would be created and by code it has been internally 
implemented to do a 80% and 20% training, test split respectively to train and validate our network on our given dataset.

The picture is captured from the webcam, where I have selected 200 frames of picture to train on for every student, hence
at the end there would be 160 images to train on and 40 images to validate our results.
So, in a fashion we are creating our dataset instantly.

After the profiles of all the students are created along with there facial capture, we then can go for a train phase.
After trained we can use to detect who the student is.

In order to avoid the painstaking task of training the network again and again, the learnt parameters of the model is saved
and can be loaded from our working directory the next session.

While recording the image data for the student, it'd be better if you turn your head around for different angle, so that
the network could be well prepared and give a better prediction.
Moreover, data augmentation is done in order to increase accuracy and decrease overfitting, plus
the participation of the dropout for the same purpose.

Why it would succeed well in this operation?
In the end, we want to stand infront of the camera and the program would detect exactly the identity.
Since, there wouldn't be any expected challenge to detect, such as, distortion we would have a straight forward approach.

========================================================================================================================================
Updates, problems discovered, how it can be better used?

Well this system harnesses the working mechanism of a traditional Convolutional Neural Network, so it would be very difficult and the accuracy wouldn't be nice to distinguish between individuals, and hence I discovered after many training sessions and parameter tunings that it wasn't doing well enough.
It'll however work pretty good in classification problems where we would have distinct features to discriminate and identify.
like cats, dogs, birds, human and so on.

However, I believe if I am able to restrict the region of my image to just the face and then train, then the CNN would learn those region and identify the features better and would as a result perform better and the computation would be very less. It's to some extent like the RCNN approach.

So, now I'll update the program for object classification, and in the next update I'll do that aforementioned biometric task.

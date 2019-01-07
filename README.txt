libraries required
h5py
keras
tensorflow
matplotlib
scikitlearn
pandas 
numpy

____________________________________________________________________________
add file from lab 2 to main directory
Add dataset folder to AMLSassignment\floyd\input
Add additional datasets in main directory
RUN remove_noise.py to get noise_classified.csv(contains names of files 
and whether its noise(0) or a face (1))

Preprocessing.py Has 2 Functions
!!In Preprocessing.py make sure image_dir is correct!! with location of dataset

Facefeatures Dlib returns the features of the face as implemented in the lab 2 Scripts
input(attribute,testsize(in %),pictureshape)
output (x_train,y_train,x_test,y_test)

preprocessingRGB (attribute1,testsize,pictureshape) Returns filenames Xtest ,ytest xtrain and ytrain
Tasks folder contain the pre maid task.csv files formatted manually
____________________________________________________________________________
 Go through each task jupyter notebook to create all csv files.
Avoid training the models saved models are provided just skip training cell and use load model cells
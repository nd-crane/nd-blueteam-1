# raite_droid_ngebm_models
We need to have our code for training models available on github for the 2023 RAITE event. This  repo is for the droid and ngebm models.

The training_code/jobs/ folder contains the submissions used to generate the models.

Requirements
-------------
The conda env I used is listed in req.txt

Location
---------
The models are at the top of the repo and are named:
droid_model.pth
ngebm_model.pth

Testing
---------
The testing_code/test_lightweight.py is a script to test a single image.
It has two inputs, imagePath and modelPath
It has one output, a single softmax score indicating if there are people or no people.
If the value is over 0.5, there are no people.
If the value is under 0.5, there are people.

Here is an example of the test code
python test_lightweight.py -imagePath ../../Data/graite/dataset/frames/mixed_singlefile_pair_standing_split1_0058.png -modelPath ../droid_model.pth

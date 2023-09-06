# raite_droid_ngebm_models
We need to have our code for training models available on github for the 2023 RAITE event. This  repo is for the droid and ngebm models.

The training_code/jobs/ folder contains the submissions used to generate the two models.

The testing_code/test_lightweight.py is a script to test a single image.
Here is an example:
python test_lightweight.py -imagePath ../../Data/graite/dataset/frames/mixed_singlefile_pair_standing_split1_0058.png -modelPath ../models_raite/densenet_droid_raite_1/Logs/droid_model.pth

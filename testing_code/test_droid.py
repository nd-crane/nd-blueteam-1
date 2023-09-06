import sys
import torch
import torchvision.models as models
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import glob
import os
import csv
import numpy as np
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
sys.path.append("../")
import random

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    device = torch.device('cpu')
    parser.add_argument('-imageFolder', default='../../Data/',type=str)
    parser.add_argument('-modelPath',  default='../model_output_paper/paper_models/densenet/cams_0.5/cams_0.51/Logs/final_model.pth',type=str)
    parser.add_argument('-csv',  default="../csvs/gan_test_sets/stylegan2.csv",type=str)
    parser.add_argument('-output_dir',  default="../model_output_paper/paper_models/densenet/cams_0.5/cams_0.51/local_results/",type=str)
    parser.add_argument('-output_filename', default="scores.csv",type=str)
    parser.add_argument('-network',  default="densenet",type=str)
    parser.add_argument('-noSample', action='store_true')
    parser.add_argument('-fakeToken', default='Spoof', type=str)
    parser.add_argument('-sampleFactor',default=0.00835,type=float)
    parser.add_argument('-realSampleFactor',default=0.167,type=float)
    parser.add_argument('-nClasses',default=2,type=int)
    args = parser.parse_args()

    os.makedirs(args.output_dir,exist_ok=True)

    # file exist?
    #if os.path.exists(os.path.join(args.output_dir,args.output_filename)):
    #    sys.exit("File exists: " + args.output_filename)  

    # seed random
    random.seed(1234)

    # Load weights of single binary DesNet121 model
    weights = torch.load(args.modelPath,map_location=torch.device('cpu'))
    if args.network == "resnet":
        im_size = 224
        model = models.resnet50(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, args.nClasses)
    elif args.network == "inception":
        im_size = 299
        model = models.inception_v3(pretrained=True,aux_logits=False)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, args.nClasses)
    else: # else DenseNet
        im_size = 224
        model = models.densenet121(pretrained=True)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, args.nClasses)
    
    model.load_state_dict(weights['state_dict'])
    model = model.to(device)
    model.eval()

    if args.network == "xception":
        pass
    else:
        # Transformation specified for the pre-processing
        transform = transforms.Compose([
                    transforms.Resize([im_size, im_size]),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])

    imagesScores=[]
    sigmoid = nn.Sigmoid()

    # imageFiles = glob.glob(os.path.join(args.imageFolder,'*.jpg'))
    imageCSV = open(args.csv,"r")
    for entry in tqdm(imageCSV):

        tokens = entry.split(",")
        if tokens[0] != 'test':
            continue

        if not args.noSample:

            # take random sample, 1/6 if fake
            if  (args.fakeToken in tokens[1] and random.random() > args.sampleFactor) or (
                 args.fakeToken not in tokens[1] and random.random() > args.realSampleFactor):
                continue

        upd_name = tokens[-1].replace("\n","")
        # upd_name = upd_name.replace(upd_name.split(".")[-1],"png")
        imgFile = args.imageFolder + upd_name

        # Read the image
        image = Image.open(imgFile).convert('RGB')
        # Image transformation
        tranformImage = transform(image)
        image.close()


        ## START COMMENTING HERE

        tranformImage = tranformImage[0:3,:,:].unsqueeze(0)
        tranformImage = tranformImage.to(device)

        # Output from single binary CNN model
        with torch.no_grad():
            output = model(tranformImage)

        if 'bird' in args.imageFolder:
            PAScore = sigmoid(output).detach().cpu().numpy()
            SMScore = nn.Softmax(dim=1)(output).detach().cpu().numpy()
            imagesScores.append([imgFile, PAScore, SMScore])
        else:
            PAScore = sigmoid(output).detach().cpu().numpy()[:, 1]
            SMScore = nn.Softmax(dim=1)(output).detach().cpu().numpy()[:, 1]
            imagesScores.append([imgFile, PAScore[0], SMScore[0]])

    # Writing the scores in the csv file
    with open(os.path.join(args.output_dir,args.output_filename),'w',newline='') as fout:
        writer = csv.writer(fout)
        writer.writerows(imagesScores)


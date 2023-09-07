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
    parser.add_argument('-imagePath', default='./image.jpg',type=str)
    parser.add_argument('-modelPath',  default='./model_final.pth',type=str)
    parser.add_argument('-output_dir',  default="./",type=str)
    parser.add_argument('-output_filename', default="scores.csv",type=str)
    parser.add_argument('-network',  default="densenet",type=str)
    parser.add_argument('-nClasses',default=2,type=int)
    args = parser.parse_args()

    os.makedirs(args.output_dir,exist_ok=True)


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

    # Transformation specified for the pre-processing
    transform = transforms.Compose([
                    transforms.Resize([im_size, im_size]),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])

    sigmoid = nn.Sigmoid()

    # Read the image
    image = Image.open(args.imagePath).convert('RGB')
    # Image transformation
    tranformImage = transform(image)
    image.close()


    tranformImage = tranformImage[0:3,:,:].unsqueeze(0)
    tranformImage = tranformImage.to(device)

    # Output from single binary CNN model
    with torch.no_grad():
        output = model(tranformImage)

        

#        PAScore = sigmoid(output).detach().cpu().numpy()[:, 1]
        SMScore = nn.Softmax(dim=1)(output).detach().cpu().numpy()[:, 1]
#        print(f'Image,PAScore,SMScore\n{args.imagePath},{PAScore[0]},{SMScore[0]}')
        print(f'{SMScore[0]}')

    # Writing the scores in the csv file
#    with open(os.path.join(args.output_dir,args.output_filename),'w',newline='') as fout:
#     fout.write(f'Image,PAScore,SMScore\n{args.imagePath},{PAScore[0]},{SMScore[0]}')

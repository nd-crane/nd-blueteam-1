import os
import sys
import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
import torchvision.models as models
import torch
import torch.nn as nn
import torch.optim as optim
from dataset_Loader_cam import datasetLoader
from tqdm import tqdm
sys.path.append("../")
# sys.path.append("./")
import random
import pdb
import time

# Description of all argument
parser = argparse.ArgumentParser()
parser.add_argument('-batchSize', type=int, default=20)
parser.add_argument('-nEpochs', type=int, default=150)
parser.add_argument('-csvPath', required=False, default= '../csvs/original.csv',type=str)
parser.add_argument('-datasetPath', required=False, default= '../../Data/',type=str)
parser.add_argument('-outputPath', required=False, default= '../model_output_local/cvpr_test/',type=str)
parser.add_argument('-heatmaps', required=False, default= '../../Data/train/heatmaps/',type=str)
parser.add_argument('-alpha_cyborg', required=False, default=0.0,type=float)
parser.add_argument('-alpha_fmmce', required=False, default=0.0,type=float)
parser.add_argument('-alpha_hbcam', required=False, default=0.0,type=float)
parser.add_argument('-alpha_droid', required=False, default=0.0,type=float)
parser.add_argument('-alpha_xent', required=False, default=0.0,type=float)
parser.add_argument('-network', default= 'densenet',type=str)
parser.add_argument('-nClasses', default= 2,type=int)
parser.add_argument('-log',action='store_true')
parser.add_argument('-keeprate', required=False, default=1.0,type=float)
parser.add_argument('-kld',action='store_true')
parser.add_argument('-lr',required=False, default=.002, type=float)

# NG-EBM args
parser.add_argument('-alpha_energy_variance', required=False, default=0.0, type=float)
parser.add_argument('-alpha_energy_derivative', required=False, default=0.0, type=float)

args = parser.parse_args()
device = torch.device('cuda')

print(args)


activation = {}
def getActivation(name):
  # the hook signature
  def hook(model, input, output):
    activation[name] = output
  return hook


# Definition of model architecture
if args.network == "resnet":
    im_size = 224
    map_size = 7
    model = models.resnet50(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, args.nClasses)
    model = model.to(device)
    model.layer4[-1].conv3.register_forward_hook(getActivation('features'))
elif args.network == "inception":
    im_size = 299
    map_size = 8
    model = models.inception_v3(pretrained=True,aux_logits=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, args.nClasses)
    model = model.to(device)  
    model.Mixed_7c.register_forward_hook(getActivation('features'))
else:
    im_size = 224
    map_size = 7
    model = models.densenet121(pretrained=True)
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, args.nClasses)
    model = model.to(device)

print(model)

# Create destination folder
os.makedirs(args.outputPath,exist_ok=True)

# Creation of Log folder: used to save the trained model
log_path = os.path.join(args.outputPath, 'Logs')
if not os.path.exists(log_path):
    os.mkdir(log_path)


# Creation of result folder: used to save the performance of trained model on the test set
result_path = os.path.join(args.outputPath , 'Results')
if not os.path.exists(result_path):
    os.mkdir(result_path)


if os.path.exists(result_path + "/DesNet121_Histogram.jpg") and os.path.exists(log_path + "/DesNet121_best.pth"):
    print("Training already completed for this setup, exiting...")
    sys.exit()


class_assgn = {'Real':0,'Synthetic':1}
if 'livdet' in args.csvPath:
    class_assgn = {'Live':0,'Spoof':1}


# Dataloader for train and test data
dataseta = datasetLoader(args.csvPath,args.datasetPath,train_test='train',c2i=class_assgn,map_location=args.heatmaps,map_size=map_size,im_size=im_size,network=args.network,keeprate=args.keeprate)
dl = torch.utils.data.DataLoader(dataseta, batch_size=args.batchSize, shuffle=True, num_workers=0, pin_memory=True)
dataset = datasetLoader(args.csvPath,args.datasetPath, train_test='test', c2i=dataseta.class_to_id,map_location=args.heatmaps,map_size=map_size,im_size=im_size,network=args.network)
test = torch.utils.data.DataLoader(dataset, batch_size=args.batchSize, shuffle=True, num_workers=0, pin_memory=True)
dataloader = {'train': dl, 'test':test}


# Description of hyperparameters
lr = args.lr
solver = optim.SGD(model.parameters(), lr=lr, weight_decay=1e-6, momentum=0.9)
lr_sched = optim.lr_scheduler.StepLR(solver, step_size=12, gamma=0.1)

criterion = nn.CrossEntropyLoss()

if args.keeprate < 1:
    criterion_hmap = nn.MSELoss(reduction='none')
else:
    criterion_hmap = nn.MSELoss(reduction='mean')

if args.kld:
    criterion_hmap = nn.KLDivLoss(reduction='batchmean')

# File for logging the training process
with open(os.path.join(log_path,'params.json'), 'w') as out:
    hyper = vars(args)
    json.dump(hyper, out)
log = {'iterations':[], 'epoch':[], 'validation':[], 'train_acc':[], 'val_acc':[], 'entropy':[],
        'class_comp':[], 'cyborg_comp':[], 'hbcam_comp':[],'fmmce_comp':[],'droid_comp':[], 
        'loss_b4_back':[], 'interm_loss':[]}



#####################################################################################
#
############### Training of the model and logging ###################################
#
#####################################################################################

edict={'energy_epoch': [],'energy_var': [],'energy_der': [], 'energy_raw': []}
train_loss=[]
test_loss=[]
bestAccuracy = 0
bestEpoch=0
train_step = 0
val_step = 0

epoch_start = 0
cam_stack = None
hmap_stack = None
cam_stack_params = None

# load last data point
# if True:
#     ckpt_dict = torch.load(os.path.join(log_path,'current_model.pth'))
#     model.load_state_dict(ckpt_dict["state_dict"])
#     solver.load_state_dict(ckpt_dict['optimizer'])
#     epoch_start = ckpt_dict['epoch']

# capture entropy information
entrops_mean = 0
entrops_mean_count = 0
max_entropy = 0.
min_entropy = 1000.

# softmax
sm = nn.Softmax(dim=1)

# loop
for epoch in range(epoch_start, epoch_start + args.nEpochs):

    for phase in ['train', 'test']:
        train = (phase=='train')
        if phase == 'train':
            model.train()
        else:
            model.eval()

        tloss = 0.
        acc = 0.
        tot = 0
        c = 0
        testPredScore = []
        testTrueLabel = []
        imgNames=[]
        with torch.set_grad_enabled(True):
            for batch_idx, (data, cls, imageName, hmap, keep) in enumerate(tqdm(dataloader[phase])):
                if data.shape[0] == 1:
                    continue

                data.requires_grad=True

                # Data and ground truth
                # setup for differentiating energy w.r.t. images
                if args.alpha_energy_derivative > 0.0:
                    data = torch.autograd.Variable(data, requires_grad=True)

                # images
                data = data.to(device)

                # labels
                cls = cls.to(device)

                # human observations
                hmap = hmap.to(device)

                # get logits
                outputs = model(data)

                # Prediction of accuracy
                pred = torch.max(outputs,dim=1)[1]
                corr = torch.sum((pred == cls).int())
                acc += corr.item()
                tot += data.size(0)
                class_loss = criterion(outputs, cls)

                # Running model over data
                # Cyborg
                if phase == 'train':
                    if args.network == "densenet":
                        # Features is the last feature map set of the network
                        # They are 7x7 maps and there are 1024 of them
                        # it has dimension [20, 1024, 7, 7] (batch size is 20)
                        features = model.features(data)

                        # parameters are the coefficients of the linear combinations of the maps for each class real/synthetic
                        # it has dimension [2, 1024]
                        params = list(model.classifier.parameters())[0]
                    elif args.network == "inception":
                        features = activation['features']
                        params = list(model.fc.parameters())[0]
                    elif args.network == "resnet":
                        features = activation['features']
                        params = list(model.fc.parameters())[0]
                    else:
                        print("INVALID ARCHITECTURE:",args.network)
                        sys.exit()

                    # Annotation for Densenet

                    # bz=20 (batch size), nc=1024 (features depth), h=w=7
                    bz, nc, h, w = features.shape

                    # make the actual maps a linear array of [49]
                    beforeDot =  features.reshape((bz, nc, h*w))

                    # create list for the cams
                    cams = []

                    # create list for entropies
                    entrops = []

                    # loop over the batch of features
                    # ids identifies each of the batch elements
                    # bd is the features for the batch image
                    for ids,bd in enumerate(beforeDot):

                        # select the weight set from the real/synthetic array per image in the batch
                        weight = params[pred[ids]]

                        # multiply the [1024] weights by the [1024,49] feature maps
                        # gives a [49] weighted sum of the feature maps
                        cam = torch.matmul(weight, bd)

                        # reshape back to the 7x7 feature map size
                        cam_img = cam.reshape(h, w)

                        # normalize the result to the range [0,1]
                        cam_img = cam_img - torch.min(cam_img)
                        cam_img = cam_img / torch.max(cam_img)

                        # add it to the list
                        cams.append(cam_img)

                        # calculate entropy for cam
                        # we need to scale as its a probability map
                        sum_M_i_j = torch.sum(cam_img)
                        cam_img_prob = cam_img / (sum_M_i_j + 1e-10) + 1e-10

                        # calculate and append entropy
                        entrop = -torch.sum(cam_img_prob * torch.log(cam_img_prob))
                        entrops.append(entrop)

                    # convert the 20 cams from a list to array
                    # should have dimentions [20, 7, 7]
                    cams = torch.stack(cams)

                    # get entropy max_min
                    e_max = max(entrops).clone().cpu().detach().numpy()
                    e_min = min(entrops).clone().cpu().detach().numpy()
                    if e_max > max_entropy:
                        max_entropy = e_max
                    if e_min < min_entropy:
                        min_entropy = e_min

                    # take mean of 20 entropies
                    entrops = torch.stack(entrops)
                    entrops_mn = torch.mean(entrops)

                    # capture for diagnostics
                    entrops_mean += entrops_mn.clone().cpu().detach().numpy()
                    entrops_mean_count += 1


                    # Cyborg losses
                    hmap_loss = 0

                    # calculate the Mean Square Error between the maps
                    # and the [20, 7, 7] resized human saliency maps
                    if args.alpha_cyborg != 0:
                        if args.keeprate<1:
                            for k in keep:
                                print(f'!grab_keepbool: {k}')
                            mse_scores = criterion_hmap(cams,hmap)
                            print(f'!grab_msesize: {mse_scores[keep].shape}')
                            cyborg_comp = args.alpha_cyborg * torch.mean(mse_scores[keep])
                        else:
                            cyborg_comp = args.alpha_cyborg * criterion_hmap(cams,hmap)
                        hmap_loss += cyborg_comp
                        log['cyborg_comp'].append(cyborg_comp.clone().cpu().detach().numpy().item())

                    # Entropy loss for DROID
                    if args.alpha_droid != 0:
                        droid_comp = args.alpha_droid * torch.mean(torch.log(entrops + 1.))
                        hmap_loss += droid_comp
                        log['droid_comp'].append(droid_comp.clone().cpu().detach().numpy().item())
                    #hmap_loss += args.alpha_droid * entrops_mn

                    # (FMMCE) Non-log cam entropy
                    if args.alpha_fmmce != 0:
                        fmmce_comp = args.alpha_fmmce * torch.mean(entrops)
                        hmap_loss += fmmce_comp
                        log['fmmce_comp'].append(fmmce_comp.clone().cpu().detach().numpy().item())

                    # (HB_CAM) CAM/human saliencey entropy loss
                    if args.alpha_hbcam != 0:
                        # calculate human salience map entropy
                        hentrops = []
                        for i in range(hmap.shape[0]):
                            hsum_M_i_j = torch.sum(hmap[i])
                            hmap_prob = hmap[i] / (hsum_M_i_j + 1e-10) + 1e-10
                            hentrop = -torch.sum(hmap_prob * torch.log(hmap_prob))
                            hentrops.append(hentrop)
                        hentrops = torch.mean(torch.stack(hentrops))
                        hbcam_comp = args.alpha_hbcam * (entrops_mn - hentrops)**2
                        hmap_loss += hbcam_comp
                        log['hbcam_comp'].append(hbcam_comp.clone().cpu().detach().numpy().item())

                    # logs
                    log['entropy'].append(entrops.clone().cpu().detach().numpy().tolist())

                else:
                    hmap_loss = 0

                # Optimization of weights for training data
                if phase == 'train':
                    # loss
                    interm_loss = 0

                    # Code for NG-EBM
                    energy = outputs.logsumexp(dim=1, keepdim=False)
                    edict['energy_epoch'].append(epoch)
                    edict['energy_raw'].append(energy.cpu().detach().numpy().tolist())

                    e_mean = torch.mean(-torch.log(entrops + 1.)) #torch.mean(torch.log(entrops))#energy)
                    energy_loss = torch.mean((e_mean - energy)**2)
                    edict['energy_var'].append(energy_loss.clone().cpu().detach().numpy().tolist())

                    # EV loss
                    if args.alpha_energy_variance > 0.0:
                        interm_loss += args.alpha_energy_variance * energy_loss #torch.mean(energy) #

                    # get gradient of energy w.r.t. images
                    if args.alpha_energy_derivative > 0.0:
                        # calculate derivatives of energfy w.r.t. images
                        f_prime = torch.autograd.grad(energy.sum(), data, create_graph=True)[0]
                        grad = f_prime.view(data.size(0), -1)
                        # Loss is the mean of the derivative norms
                        dlogpx_dx_loss = torch.mean(grad.norm(p=2, dim=1))
                        edict['energy_der'].append(dlogpx_dx_loss.clone().cpu().detach().numpy().tolist())

                        # ED loss
                        interm_loss += args.alpha_energy_derivative * dlogpx_dx_loss
                    #log['interm_loss'].append(interm_loss.clone().cpu().detach().numpy().tolist())

                    # combine CAM and x-entropy loss
                    if args.log:
                        loss = (args.alpha_xent)*(class_loss) + (torch.log(1+hmap_loss))
                    else:
                        loss = (args.alpha_xent)*(class_loss) + (hmap_loss)
                    log['class_comp'].append(class_loss.clone().cpu().detach().numpy().item())

                    # add in energy losses
                    loss += interm_loss
                    log['loss_b4_back'].append(loss.item())

                    train_step += 1
                    solver.zero_grad()
                    loss.backward()
                    solver.step()
                    log['iterations'].append(loss.item())
                elif phase == 'test':
                    loss = class_loss #criterion(outputs, cls)      #

                    val_step += 1
                    temp = outputs.detach().cpu().numpy()
                    scores = np.stack((temp[:,0], np.amax(temp[:,1:args.nClasses], axis=1)), axis=-1)
                    testPredScore.extend(scores)
                    testTrueLabel.extend((cls.detach().cpu().numpy()>0)*1)
                    imgNames.extend(imageName)

                tloss += loss.item()
                c += 1

        # Logging of train and test results
        if phase == 'train':
            log['epoch'].append(tloss/c)
            log['train_acc'].append(acc/tot)
            if entrops_mean_count > 0:
                print('Epoch: ', epoch, 'Train loss: ',tloss/c, 'Accuracy: ', acc/tot, \
                        'Entropy', entrops_mean / entrops_mean_count, \
                         'E max', max_entropy, 'E min', min_entropy)
            else:
                print('Epoch: ', epoch, 'Train loss: ',tloss/c, 'Accuracy: ', acc/tot, \
                     'E max', max_entropy, 'E min', min_entropy)

            # reset entropy values
            entrops_mean_count = 0
            entrops_mean = 0

            max_entropy = 0.
            min_entropy = 1000.

            train_loss.append(tloss / c)

        elif phase == 'test':
            log['validation'].append(tloss / c)
            log['val_acc'].append(acc / tot)
            print('Epoch: ', epoch, 'Test loss:', tloss / c, 'Accuracy: ', acc / tot)
            lr_sched.step()
            test_loss.append(tloss / c)
            accuracy = acc / tot
            if (accuracy >= bestAccuracy):
                bestAccuracy =accuracy
                testTrueLabels = testTrueLabel
                testPredScores = testPredScore
                bestEpoch = epoch
                save_best_model = os.path.join(log_path,'final_model.pth')
                states = {
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optimizer': solver.state_dict(),
                }
                torch.save(states, save_best_model)
                testImgNames= imgNames

    states = {
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'optimizer': solver.state_dict(),
    }
    with open(os.path.join(log_path,'model_log.json'), 'w') as out:
        json.dump(log, out)
    torch.save(states, os.path.join(log_path,f'current_model.pth'))

with open(os.path.join(log_path,'energy_log.json'),'w') as out:
    json.dump(edict, out)


# Plotting of train and test loss
plt.figure()
plt.xlabel('Epoch Count')
plt.ylabel('Loss')
plt.plot(np.arange(0, args.nEpochs), train_loss[:], color='r')
plt.plot(np.arange(0, args.nEpochs), test_loss[:], 'b')
plt.legend(('Train Loss', 'Validation Loss'), loc='upper right')
plt.savefig(os.path.join(result_path,'model_Loss.jpg'))


print('TRAIN COMPLETE')


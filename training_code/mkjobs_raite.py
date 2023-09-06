import os
if not os.path.exists('./jobs'):
    os.mkdir('./jobs')

dataset='raite'
folder_name='raite'


with open(f'jobs/sub_{folder_name}.sh','w') as subF:
    subF.write('#!/bin/bash/\n')
    for model in ['densenet']:
        for exp in ['droid','ngebm']:
            for run in range(1,2):

                runname=f'{model}_{exp}_{dataset}_{run}'

                with open(f'jobs/train_{runname}.job','w') as outF:
                    text="\n".join([f'#!/bin/bash',
                                    f'#$ -l gpu=1',
                                    f'#$ -q gpu',
                                    f'module load conda',
                                    f'conda activate cyborg2',
                                   # f'pip3 install torchvision --user',
                                    f'\n',
                                    f'cd ..',
                                    f''])
                    if exp=='ngebm':
                        extraParams='-alpha_xent 0.5 -alpha_energy_derivative 0.5'
                    elif exp=='droid':
                        extraParams='-alpha_xent 0.5 -alpha_droid 0.5'
                    else:
                        raise("typo in setting extraParams")
                    if model == 'xception':
                        extraParams+=' -batchSize 10'

                    extraParams+=' -nEpochs 50'


                    text="\n".join([text,
                                    f'python train_droid_alphas.py -datasetPath ../../Data/graite/dataset/frames/ -outputPath ../models_{folder_name}/{runname}/ -network {model} -csvPath ../csvs/train_classification_raite.csv {extraParams}'])
                    outF.write(text)
                    subF.write(f'qsub train_{runname}.job\n')

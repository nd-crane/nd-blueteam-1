# RAITE: ND Blue Team 1


This repository contains Notre Dame Blue Team's solutions for RAITE 2023.
It provides scripts for training and testing two models, `droid` and `ngebm`, to classify images based on the presence of humans. These models are trained to detect whether an image contains a human or not.


## Prerequisites

- [Anaconda](https://www.anaconda.com/products/distribution) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)

## Setup

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/nd-crane/nd-blueteam-1.git
   cd nd-blueteam-1
   ```

2. **Set Up the Anaconda Environment**:
   Create a new Anaconda environment using the provided `environment.yml` file:
   ```bash
   conda env create -f environment.yml
   ```

   Activate the environment:
   ```bash
   conda activate nd-blueteam-1
   ```

3. **Download the Models**:
   The models (`droid` and `ngebm`) can be downloaded from the provided links.

   - [Download Droid Model Here](https://drive.google.com/file/d/1xn6oMd0DEU7Ib6TXgLKgyeKOn9zjiXQj/view?usp=sharing)
   - [Download Ngebm Model Here](https://drive.google.com/file/d/1T7th1PsvMMetXPxq3o6NswhXdfypMcBk/view?usp=sharing)

## Training

### Droid Model
To train the `droid` model, use the following command:

```bash
python training_code/train_droid_alphas.py -datasetPath [PATH_TO_DATASET] \
    -outputPath [PATH_TO_SAVE_MODEL] \
    -network densenet \
    -alpha_xent 0.5 \
    -alpha_droid 0.5 \
    -nEpochs 50
```

### Ngebm Model
To train the `ngebm` model, use the following command:

```bash
python training_code/train_droid_alphas.py -datasetPath [PATH_TO_DATASET] \
    -outputPath [PATH_TO_SAVE_MODEL] \ 
    -network densenet \
    -alpha_xent 0.5 \
    -alpha_energy_derivative 0.5 \
    -nEpochs 50
```


## Testing

To test an image using a specific model, use the following command:

```bash
python testing_code/test_lightweight.py \
    -modelPath [PATH_TO_YOUR_MODEL]
    -imagePath [PATH_TO_YOUR_IMAGE] \
```

**Example**:
```bash
python testing_code/test_lightweight.py \
    -modelPath ../droid_model.pth \
    -imagePath mixed_singlefile_pair_standing_split1_0058.png \

```


## Results Interpretation

The script will output a single softmax score indicating the presence of people in the image:

- If the value is **over 0.5**, there are no people.
- If the value is **under 0.5**, there are people.

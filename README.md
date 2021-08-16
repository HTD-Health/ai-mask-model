# Model

## Data preparation

1. Download MaskedFace-Net dataset from https://github.com/cabani/MaskedFace-Net
2. Move the content to get the following directory structure:

```
├── ai-mask-model
│   ├── data                        <- All your data modules should be located here!
│   │   ├── dataframe               <- Place to store JSON with mapping data calsses
│   │   │
│   │   └── dataset                 <- Whole dataset
│   │       ├── test                
│   │       │   ├── 0               <- Part of the dataset containing unproperly masked faces
│   │       │   │   ├── 00001.jpg   <- Image
│   │       │   │   ├── 00002.jpg
│   │       │   │   └── ...
│   │       │   └── 1               <- Part of the dataset containing properly masked faces
│   │       │       ├── 00001.jpg   <- Image
│   │       │       ├── 00002.jpg
│   │       │       └── ...
│   │       ├── train
│   │       │   ├── 0               <- Part of the dataset containing unproperly masked faces
│   │       │   │   ├── 00001.jpg   <- Image
│   │       │   │   ├── 00002.jpg
│   │       │   │   └── ...
│   │       │   └── 1               <- Part of the dataset containing properly masked faces
│   │       │       ├── 00001.jpg   <- Image
│   │       │       ├── 00002.jpg
│   │       │       └── ...
│   │       └── valid
│   │           ├── 0               <- Part of the dataset containing unproperly masked faces
│   │           │   ├── 00001.jpg   <- Image
│   │           │   ├── 00002.jpg
│   │           │   └── ...
│   │           └── 1               <- Part of the dataset containing properly masked faces
│   │               ├── 00001.jpg   <- Image
│   │               ├── 00002.jpg
│   │               └── ...
│   └── ...
└── ...

```

## How to run

First, install dependencies

```bash
# clone project
git clone https://github.com/HTD-Health/ai-smart-mirror.git

# install project
cd ai-smart-mirror
pip install -r requirements.txt
```

Next, just train the model by running:

```bash
python train.py data/datasets/
```

When you finish training your network, to use it just type:

```bash
python predict.py path_to_image_to_check.jpg
```

## Raspberry Pi 3

### Remote access

We are using `Dataplicity` for remote access to Raspberry Pi.
If you want to have access to Raspberry Pi for test purposes send your email address and request to be added to [Jędrzej Polaczek](https://github.com/jedrzejpolaczek).

### How to use

#### Repository

Repository is under path `home/pi/Workplace/ai-smart-mirror`

#### Making changes

To make changes (like changing branch) you need to run the command as `superuser`.
To make it be you need to type `su pi` (and execute this command) and after that enter the appropriate password.
Quick guide: `https://docs.dataplicity.com/docs/superuser`

### How to install
To install this library just run in `ai-mask-model` folder command `pip install -e .` when you are in the right virtual environment.
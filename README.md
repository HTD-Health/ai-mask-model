# ai-smart-mirror

## Data preparation

1. Download MaskedFace-Net dataset from https://github.com/cabani/MaskedFace-Net
2. Move the content to get the following directory structure:

```
├── mask_detector
│   ├── data                        <- All your data modules should be located here!
│   │   ├── dataset                 <- Whole dataset
│   │   │   ├── mask                <- Part of the dataset containing properly masked faces
│   │   │   │   ├── 00000           <- Dataset chunk containing images
│   │   │   │   │   ├── 00001.jpg   <- Image
│   │   │   │   │   ├── 00002.jpg
│   │   │   │   │   └── ...
│   │   │   │   ├── 01000
│   │   │   │   └── ...
│   │   │   └── no_mask             <- Part of the dataset containing incorrectly masked faces
│   │   │       ├── 00000           <- Dataset chunk containing images
│   │   │       │   ├── 00001.jpg   <- Image
│   │   │       │   ├── 00002.jpg
│   │   │       │   └── ...
│   │   │       ├── 01000
│   │   │       └── ...
│   │   └── dataframe               <- Place to store generated dataframes
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

Next, navigate to any file and run it.

```bash
# module folder
cd mask_detector

# run module
python mask_classifier.py
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

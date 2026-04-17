# ScribThyroidSAM
## Installation
1. Create a virtual environment `conda create -n scribthyroidsam python=3.10 -y` and activate it `conda activate scribthyroidsam`
2. Enter the MedSAM folder `cd MedSAM` and run `pip install -e .`
3. Install other dependencies
After MedSAM is installed, install the remaining required packages: `pip install -r requirements.txt`

## Data Preparation
The DDTI and TN3K datasets can be downloaded from the following links:
- DDTI: https://github.com/XXX/DDTI
- TN3K: https://github.com/XXX/TN3K
- Please prepare the DDTI and TN3K datasets according to the project directory structure before training.
Example:

data/
├── DDTI/
│   ├── images/
│   ├── masks/
│   └── scribbles/
└── TN3K/
    ├── images/
    ├── masks/
    └── scribbles/

## Train
To train the model on the DDTI dataset, run:

bash
python train.py --dataset DDTI --config configs/ddti.yaml
python train.py --dataset TN3K --config configs/tn3k.yaml

If you want to train on a specific dataset or use a custom configuration, please modify the corresponding dataset path and training settings in the configuration file before running the command.

The trained checkpoints and logs will be saved to the output directory specified in the configuration.

## Test
To train the model on the DDTI dataset, run:
python test.py --dataset DDTI --config configs/ddti.yaml --checkpoint path/to/checkpoint.pth

To evaluate on the TN3K dataset, run:
python test.py --dataset TN3K --config configs/tn3k.yaml --checkpoint path/to/checkpoint.pth

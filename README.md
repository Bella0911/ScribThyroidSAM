# ScribThyroidSAM

## Installation

1. Create a virtual environment and activate it:

```bash
conda create -n scribthyroidsam python=3.10 -y
conda activate scribthyroidsam

2. Enter the `MedSAM` folder and install it:

```bash
cd MedSAM
pip install -e .
```

3. Install the remaining dependencies:

```bash
pip install -r requirements.txt
```
## Download Checkpoints

Please download the required pretrained weights before training and testing.

### MedSAM Weights

Download the MedSAM pretrained weights medsam_vit_b.pth from the official source:

- MedSAM: https://github.com/bowang-lab/MedSAM

After downloading, place the checkpoint file in the following directory:

```text
MedSAM/work_dir/
````

---

### SegFormer Weights

Download the pretrained SegFormer weights mit_b2.pth from:

* SegFormer: [https://github.com/NVlabs/SegFormer](https://github.com/NVlabs/SegFormer)

After downloading, place the checkpoint file in the following directory:

```text
weights/
```



## Data Preparation

The DDTI and TN3K datasets can be downloaded from the following links:

* [DDTI](https://drive.google.com/file/d/1q5kY51I0OQOD9Pd7_oUKw5lGww47ltwP/view?usp=drive_link)
* [TN3K](https://drive.google.com/file/d/12mLXyauJlACOaDk-Ds85fM6f8bE49QfW/view?usp=drive_link)

Please prepare the DDTI and TN3K datasets according to the following project directory structure before training:

```text
data/
├── DDTI/
│   ├── images/
│   ├── masks/
│   └── scribbles/
└── TN3K/
    ├── images/
    ├── masks/
    └── scribbles/
```

## Train

To train the model on the DDTI dataset, run:

```bash
python train.py --dataset DDTI --config configs/ddti.yaml
```

To train the model on the TN3K dataset, run:

```bash
python train.py --dataset TN3K --config configs/tn3k.yaml
```

If you want to train on a specific dataset or use a custom configuration, please modify the corresponding dataset path and training settings in the configuration file before running the command.

The trained checkpoints and logs will be saved to the output directory specified in the configuration.

## Test

To evaluate the model on the DDTI dataset, run:

```bash
python test.py --dataset DDTI --config configs/ddti.yaml --checkpoint path/to/checkpoint.pth
```

To evaluate the model on the TN3K dataset, run:

```bash
python test.py --dataset TN3K --config configs/tn3k.yaml --checkpoint path/to/checkpoint.pth
```

Before testing, please make sure that the checkpoint path and dataset path are correctly set.


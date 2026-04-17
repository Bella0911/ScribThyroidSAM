import torch

CFG = {
    'project_name': 'Thyroid',
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',

    'paths': {
        'root': 'datasets/DDTI',
        'images': 'datasets/DDTI/images',
        'scribbles': 'datasets/DDTI/scribbles',
        'masks': 'datasets/DDTI/masks',
        'embeddings': 'datasets/DDTI/medsam_embeddings',
        'train_txt': 'datasets/DDTI/train.txt',
        'val_txt': 'datasets/DDTI/test.txt',
        'work_dir': 'work_dirs/edge_aware_final',
    },

    'medsam': {
        'checkpoint': '/home/heyan/thyroid/Weakthroidsam/MedSAM/work_dir/MedSAM/medsam_vit_b.pth',
        'num_perturbations': 10,
        'min_expansion_pixels': 15,
    },
    'experiment': {
        'box_shrink_ratio': 1.0,
    },
    'train': {
        'img_size': 256,
        'batch_size': 8,
        'lr': 1e-4,
        'total_epochs': 100,
        'warmup_epochs': 20,
        'update_interval': 5,
        'ema_decay': 0.99,
        'ema_warmup_decay': 0.99,
    },
    'update': {
        'recall_threshold': 0.5,
        'use_plateau_lock': True,
        'plateau_lock_start_epoch': 20,
        'plateau_score_threshold': 0.80,
        'plateau_patience': 2,

        # Q = λ1*S_th + λ2*S_temp - λ3*D_prob
        'lambda_th': 0.45,
        'lambda_temp': 0.45,
        'lambda_prob': 0.10,
    }
}
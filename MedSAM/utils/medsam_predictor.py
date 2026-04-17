import torch
import numpy as np
from skimage import transform
import torch.nn.functional as F
from segment_anything import sam_model_registry


class MedSAMPredictor:
    def __init__(self, model_path, device='cuda'):
        """
        Initializes the MedSAM model and puts it in evaluation mode.
        """
        self.device = torch.device(device)
        self.model = sam_model_registry["vit_b"](checkpoint=model_path).to(self.device)
        self.model.eval()
        self.image_embedding = None
        self.original_size = None
        self.input_size = None
        print("MedSAM model loaded and in evaluation mode.")

    @torch.no_grad()
    def set_image(self, image: np.ndarray):
        """
        Encodes the image and stores the embedding for subsequent predictions.
        This is the "single-encode" part of the strategy.

        :param image: Input image as a numpy array (H, W, C).
        """
        self.original_size = image.shape[:2]

        # 1. Preprocess the image (resize and normalize)
        img_1024 = transform.resize(
            image, (1024, 1024), order=3, preserve_range=True, anti_aliasing=True
        ).astype(np.uint8)

        img_1024_norm = (img_1024 - img_1024.min()) / np.clip(
            img_1024.max() - img_1024.min(), a_min=1e-8, a_max=None
        )

        # 2. Convert to tensor and move to device
        img_tensor = (
            torch.tensor(img_1024_norm)
                .float()
                .permute(2, 0, 1)
                .unsqueeze(0)
                .to(self.device)
        )
        self.input_size = img_tensor.shape[2:]

        # 3. Generate and store the image embedding
        self.image_embedding = self.model.image_encoder(img_tensor)

    @torch.no_grad()
    def predict_with_box(self, box: np.ndarray) -> np.ndarray:
        """
        Performs inference using a single box prompt, reusing the stored image embedding.
        This is the "multi-decode" part.

        :param box: A bounding box in [x_min, y_min, x_max, y_max] format.
        :return: A binary segmentation mask as a numpy array.
        """
        if self.image_embedding is None:
            raise RuntimeError("Image embedding not set. Call `set_image` before `predict_with_box`.")

        H, W = self.original_size

        # Scale box from original image size to 1024x1024
        box_1024 = box / np.array([W, H, W, H]) * 1024

        box_torch = torch.as_tensor(box_1024, dtype=torch.float, device=self.device)
        box_torch = box_torch[None, None, :]  # Shape: (B, 1, 4)

        # Generate mask
        sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
            points=None, boxes=box_torch, masks=None
        )
        low_res_logits, _ = self.model.mask_decoder(
            image_embeddings=self.image_embedding,
            image_pe=self.model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )

        # Upsample mask to original image size
        low_res_pred = F.interpolate(
            torch.sigmoid(low_res_logits),
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        )

        return (low_res_pred.squeeze().cpu().numpy() > 0.5).astype(np.uint8)

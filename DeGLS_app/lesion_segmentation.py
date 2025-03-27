import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from utils.model_utils import GAUNet  # Import the GAUNet model architecture

class GAUNetSegmenter:
    def __init__(self, model_path, device='cpu', in_channels=3, out_channels=1):
        """
        Initialize the GAUNetSegmenter with model path, device, input channels, and output channels.
        """
        # Instantiate the GAUNet model architecture
        self.model = GAUNet(in_channels=in_channels, out_channels=out_channels)  # Pass in_channels and out_channels
        # Load the model's state dictionary (weights) into the architecture
        self.model.load_state_dict(torch.load(model_path, map_location=device))  # Load the weights
        self.model.eval()  # Set model to evaluation mode
        self.device = device

    def segment(self, image_path):
        # Open the image
        image = Image.open(image_path).convert("RGB")
        
        # Define the image transformations for preprocessing
        test_image_transforms = transforms.Compose([
            transforms.Resize((512, 512)),  # Resize to match training size
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Same normalization as training
        ])
        
        # Apply the transformations to the image
        image = test_image_transforms(image)
        
        # Add batch dimension since the model expects a batch of images
        image = image.unsqueeze(0).to(self.device)  # Shape: [1, 3, 512, 512]

        # Perform inference
        with torch.no_grad():
            output = self.model(image)
            probabilities = torch.sigmoid(output).squeeze().cpu().numpy()  # Get probability heatmap
            threshold = 0.8  # Define a threshold for binarization
            pred_mask = (probabilities > threshold).astype(np.float32)  # Apply threshold to get binary mask

        return pred_mask

    def save_mask(self, mask, save_path):
        # Convert the mask to an image (0-255 range)
        mask = (mask * 255).astype(np.uint8)  # Convert mask to uint8 format in the range [0, 255]
        mask_image = Image.fromarray(mask)  # Convert numpy array to PIL Image
        
        # Save the mask as an image
        mask_image.save(save_path)

from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2

class YOLOSegmenter:
    def __init__(self, model_path):
        # Load the YOLOv8 model from the specified path
        self.model = YOLO(model_path)  # Loads the model from yolo.pt
        self.model.eval()  # Set the model to evaluation mode

    def segment_and_classify(self, image_path):
        # Load the image for prediction
        image = Image.open(image_path)

        # Run inference using the YOLOv8 model
        results = self.model(image)

        # Extract the segmented leaf region and disease classification
        segmented_leaf, disease_class = self.extract_segmented_leaf(results), self.classify_disease(results)

        if disease_class == "corn_gls":
            disease_class = "Gray Leaf Spot"
        elif disease_class == "corn_nlb":
            disease_class = "Northern Leaf Blight"
        elif disease_class == "corn_rust":
            disease_class = "Common Rust"

        return segmented_leaf, disease_class

    def extract_segmented_leaf(self, results):
        # Ensure the results are valid and masks exist
        result = results[0]  # Get the first result in the list
        if hasattr(result, 'masks') and result.masks is not None:
            # Use the image (input) itself for generating the segmentation mask with color
            return self.process_masks(result.masks, result.orig_img)  # Extract the original image from the result
        else:
            raise ValueError("No segmentation masks found in results.")

    def classify_disease(self, results):
        # Extract classification information from the results
        result = results[0]  # Get the first result in the list
        if hasattr(result, 'boxes') and result.boxes is not None:
            class_idx = result.boxes.cls[0].item()  # Get the first class index (disease type)
            return result.names[class_idx]  # Map it to the class name
        else:
            raise ValueError("No classification found in results.")

    def process_masks(self, masks, orig_img):
        # Convert original image to numpy (PIL to RGB format)
        orig_img_array = np.array(orig_img)

        # Create an empty mask image with the same size as the original image
        mask_image = np.zeros_like(orig_img_array, dtype=np.uint8)

        # Iterate through all detected masks (leaf regions)
        for mask in masks.xy:
            points = np.array(mask).reshape((-1, 1, 2)).astype(np.int32)
            cv2.fillPoly(mask_image, [points], (255, 255, 255))  # Fill the mask region with white (255)

        # Apply the mask to the original image to preserve the leaf color and set the background to black
        masked_image = orig_img_array.copy()  # Make a copy of the image array
        masked_image[mask_image == 0] = 0  # Set the background to black

        # Convert the resulting masked image from BGR to RGB (if necessary)
        if masked_image.shape[2] == 3:  # Check if it's a color image
            masked_image = cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

        # Convert the resulting masked image back to a PIL image for saving or further processing
        return Image.fromarray(masked_image)

    def save_segmented_leaf(self, segmented_leaf, save_path):
        # Save the segmented leaf (the result is a PIL Image)
        segmented_leaf.save(save_path)

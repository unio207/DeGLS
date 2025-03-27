import numpy as np
import cv2

def calculate_disease_severity(segmented_leaf_path, lesion_mask_path):
  """
  Calculate the percent severity of disease on a leaf.

  Parameters:
  - segmented_leaf_path (str): File path to the color image of the segmented leaf.
  - lesion_mask_path (str): File path to the binary mask of the lesions (1 for lesion, 0 for non-lesion).

  Returns:
  - float: Percent severity of the disease on the leaf.
  """
  # Read the images from the file paths
  segmented_leaf = cv2.imread(segmented_leaf_path)
  lesion_mask = cv2.imread(lesion_mask_path, cv2.IMREAD_GRAYSCALE)

  if segmented_leaf is None:
    raise FileNotFoundError(f"Segmented leaf image not found: {segmented_leaf_path}")
  if lesion_mask is None:
    raise FileNotFoundError(f"Lesion mask image not found: {lesion_mask_path}")

  # Resize both inputs to 256x256
  segmented_leaf = cv2.resize(segmented_leaf, (256, 256), interpolation=cv2.INTER_AREA)
  lesion_mask = cv2.resize(lesion_mask, (256, 256), interpolation=cv2.INTER_AREA)

  # Binarize the segmented leaf by counting non-black pixels
  segmented_leaf = np.where(np.any(segmented_leaf > 0, axis=-1), 1, 0)

  # Ensure the lesion mask is binary
  lesion_mask = np.where(lesion_mask > 0, 1, 0)

  # Calculate the total pixel count of the segmented leaf
  total_leaf_pixels = np.sum(segmented_leaf)

  # Calculate the total pixel count of the lesions, but only within the leaf area
  total_lesion_pixels = np.sum(lesion_mask * segmented_leaf)

  # Avoid division by zero
  if total_leaf_pixels == 0:
    raise ValueError("Segmented leaf contains no pixels.")

  # Calculate the percent severity
  percent_severity = (total_lesion_pixels / total_leaf_pixels) * 100

  return round(percent_severity, 2)


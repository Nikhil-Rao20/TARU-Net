"""
3D Medical Image Segmentation and Analysis Pipeline.

This script performs the following steps:
1.  Downloads the Medical Segmentation Decathlon lung dataset from Kaggle.
2.  Loads a pre-trained U-Net model for lung segmentation.
3.  Preprocesses a target NIfTI image and its corresponding ground truth mask.
4.  Runs inference on the image slices to generate a 3D segmentation mask.
5.  Calculates the Dice Similarity Coefficient (DSC) to evaluate the prediction.
6.  Generates 3D mesh files (.stl) from the predicted and ground truth masks.
7.  Converts the .stl meshes into point clouds (.xyz).
8.  Performs surface reconstruction on the point cloud.
9.  (Optional) Colors a mesh based on Hounsfield Unit (HU) values from a
    separate CT scan to visualize tissue density.

Required packages:
- kagglehub, nilearn, niwidgets, numpy-stl, open3d, k3d, ipympl, trimesh
- torch, torchvision, scikit-image, scikit-learn, nibabel, opencv-python, tqdm
- matplotlib
"""

# =============================================================================
# 1. IMPORTS
# =============================================================================

# --- Standard Library Imports ---
import os
import glob
from typing import Tuple, List

# --- Third-party Imports ---
import cv2
import kagglehub
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import open3d as o3d
import torch
import torch.nn.functional as F
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.ndimage import binary_closing, binary_dilation, binary_erosion
from skimage import measure
from stl import mesh as stl_mesh
from tqdm.auto import tqdm

# --- Local Application/Library Specific Imports ---
from Pnuemonia_and_Cancer_Reconstruction.Models.unet_model import UNet1
from rbf_helper_functions import *


# =============================================================================
# 2. CONSTANTS AND CONFIGURATION
# =============================================================================
# --- Model and Device Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = '/content/U-Net_best_val_loss.pt'

# --- Dataset and File Configuration ---
KAGGLE_DATASET_PATH = "luumsk/medical-segmentation-decathlon-lung"
TARGET_IMAGE_FILE = 'lung_003.nii'
TARGET_MASK_FILE = 'lung_003.nii'
OUTPUT_DIR = './output_results'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Image Processing Configuration ---
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
HOUNSFIELD_MIN = -1000
HOUNSFIELD_MAX = 2000
HOUNSFIELD_RANGE = HOUNSFIELD_MAX - HOUNSFIELD_MIN


# =============================================================================
# 3. HELPER FUNCTIONS
# =============================================================================

def preprocess_nifti_volume(img_path: str, mask_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads, normalizes, and resizes NIfTI image and mask volumes slice by slice.

    Args:
        img_path (str): Path to the input NIfTI image file.
        mask_path (str): Path to the input NIfTI mask file.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing the preprocessed
        3D image array and 3D mask array.
    """
    img_nii = nib.load(img_path).get_fdata().astype(np.float32)
    lbl_nii = nib.load(mask_path).get_fdata().astype(np.int64)

    if img_nii.shape != lbl_nii.shape:
        raise ValueError("Image and mask shapes do not match.")

    num_slices = img_nii.shape[2]
    image_data = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, num_slices))
    mask_data = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, num_slices))

    print("Preprocessing NIfTI volume...")
    for i in tqdm(range(num_slices)):
        img_slice = img_nii[:, :, i]
        lbl_slice = lbl_nii[:, :, i]

        # Normalize slice to [0, 1]
        img_slice = (img_slice - img_slice.min()) / (img_slice.max() - img_slice.min() + 1e-8)

        # Convert to tensor, add channel dim, and resize
        img_tensor = torch.from_numpy(img_slice).unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, H, W)
        mask_tensor = torch.from_numpy(lbl_slice).float().unsqueeze(0).unsqueeze(0) # Shape: (1, 1, H, W)

        img_resized = F.interpolate(img_tensor, size=(IMAGE_HEIGHT, IMAGE_WIDTH), mode='bilinear', align_corners=False)
        mask_resized = F.interpolate(mask_tensor, size=(IMAGE_HEIGHT, IMAGE_WIDTH), mode='nearest')

        image_data[:, :, i] = img_resized.squeeze().numpy()
        mask_data[:, :, i] = mask_resized.squeeze().numpy()

    return image_data, mask_data

def compute_dice_score(prediction: np.ndarray, ground_truth: np.ndarray) -> float:
    """
    Calculates the Dice Similarity Coefficient (DSC) between two binary masks.

    Args:
        prediction (np.ndarray): The predicted binary mask.
        ground_truth (np.ndarray): The ground truth binary mask.

    Returns:
        float: The Dice score, ranging from 0.0 to 1.0.
    """
    pred_binary = (prediction > 0.5).astype(np.float32)
    truth_binary = (ground_truth > 0.5).astype(np.float32)

    intersection = np.sum(pred_binary * truth_binary)
    sum_of_masks = np.sum(pred_binary) + np.sum(truth_binary)

    if sum_of_masks == 0:
        return 1.0  # Conventionally, dice is 1 if both masks are empty.

    dice = (2. * intersection) / sum_of_masks
    return dice

def create_stl_mesh(vertices: np.ndarray, faces: np.ndarray) -> stl_mesh.Mesh:
    """
    Converts vertices and faces into an STL mesh object.

    Args:
        vertices (np.ndarray): Array of vertex coordinates.
        faces (np.ndarray): Array of face indices.

    Returns:
        stl_mesh.Mesh: An STL mesh object.
    """
    stl_obj = stl_mesh.Mesh(np.zeros(faces.shape[0], dtype=stl_mesh.Mesh.dtype))
    for i, f in enumerate(faces):
        for j in range(3):
            stl_obj.vectors[i][j] = vertices[f[j], :]
    return stl_obj

def plot_and_save_3d_volume(volume: np.ndarray, title: str, output_dir: str):
    """
    Generates a 3D mesh plot from a volume using marching cubes and saves it.

    Args:
        volume (np.ndarray): The 3D numpy array to plot.
        title (str): The title for the plot and the output filename.
        output_dir (str): Directory to save the plot image.
    """
    print(f"Generating 3D plot for: {title}")
    try:
        # Marching cubes algorithm to find surface
        verts, faces, _, _ = measure.marching_cubes(volume, level=0.5)
    except (ValueError, RuntimeError) as e:
        print(f"Could not perform marching cubes for {title}: {e}")
        print("This may happen if the volume is empty or constant.")
        return

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Create a 3D patch collection and add it to the plot
    mesh_collection = Poly3DCollection(verts[faces])
    mesh_collection.set_edgecolor('k')
    ax.add_collection3d(mesh_collection)

    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_zlabel("Z-axis")
    ax.set_title(title)

    # Set axes limits to match the volume dimensions
    ax.set_xlim(0, volume.shape[2])
    ax.set_ylim(0, volume.shape[1])
    ax.set_zlim(0, volume.shape[0])
    
    # Save the figure
    output_path = os.path.join(output_dir, f"{title}.png")
    plt.savefig(output_path)
    plt.close(fig) # Close figure to free memory
    print(f"Saved 3D plot to {output_path}")


# =============================================================================
# 4. MAIN EXECUTION
# =============================================================================

def main():
    """Main function to run the entire pipeline."""

    # --- 1. SETUP: LOAD MODEL AND DATASET ---
    print(f"Using device: {DEVICE}")
    model = UNet1().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    print(f"Downloading dataset: {KAGGLE_DATASET_PATH}")
    dataset_base_path = kagglehub.dataset_download(KAGGLE_DATASET_PATH)
    
    # Construct full paths to data
    image_dir = os.path.join(dataset_base_path, 'imagesTr')
    mask_dir = os.path.join(dataset_base_path, 'labelsTr')
    target_image_path = os.path.join(image_dir, TARGET_IMAGE_FILE)
    target_mask_path = os.path.join(mask_dir, TARGET_MASK_FILE)

    # --- 2. PREPROCESSING ---
    target_image_vol, ground_truth_mask_vol = preprocess_nifti_volume(
        target_image_path, target_mask_path
    )

    # --- 3. INFERENCE ---
    predicted_mask_vol = np.zeros_like(target_image_vol)
    
    print("Running model inference on image slices...")
    with torch.no_grad():
        for idx in tqdm(range(target_image_vol.shape[2])):
            img_slice = target_image_vol[:, :, idx]
            # Reshape for model: [Batch, Channel, Height, Width]
            img_tensor = torch.from_numpy(img_slice).unsqueeze(0).unsqueeze(0).float().to(DEVICE)
            
            output = model(img_tensor)
            prediction_slice = torch.sigmoid(output).squeeze().cpu().numpy()
            
            predicted_mask_vol[:, :, idx] = (prediction_slice > 0.5).astype(np.int64)

    # --- 4. EVALUATION ---
    dice_score = compute_dice_score(predicted_mask_vol, ground_truth_mask_vol)
    print(f"\nDice Score: {dice_score:.4f}")

    # --- 5. 3D MESH GENERATION AND VISUALIZATION ---
    plot_and_save_3d_volume(predicted_mask_vol, "Predicted_Mask_3D", OUTPUT_DIR)
    plot_and_save_3d_volume(ground_truth_mask_vol, "Ground_Truth_Mask_3D", OUTPUT_DIR)

    # Generate vertices and faces for STL saving
    pred_verts, pred_faces, _, _ = measure.marching_cubes(predicted_mask_vol, level=0.5)
    mask_verts, mask_faces, _, _ = measure.marching_cubes(ground_truth_mask_vol, level=0.5)

    print(f"Predicted Mask: {len(pred_verts)} vertices, {len(pred_faces)} faces")
    print(f"Ground Truth Mask: {len(mask_verts)} vertices, {len(mask_faces)} faces")

    # Save meshes as STL files
    pred_mesh_stl = create_stl_mesh(pred_verts, pred_faces)
    pred_stl_path = os.path.join(OUTPUT_DIR, 'predicted_mask.stl')
    pred_mesh_stl.save(pred_stl_path)
    print(f"Saved predicted mesh to {pred_stl_path}")
    
    gt_mesh_stl = create_stl_mesh(mask_verts, mask_faces)
    gt_stl_path = os.path.join(OUTPUT_DIR, 'ground_truth_mask.stl')
    gt_mesh_stl.save(gt_stl_path)
    print(f"Saved ground truth mesh to {gt_stl_path}")
    
    # --- 6. POINT CLOUD CONVERSION ---
    print("\nConverting STL files to Point Clouds (.xyz)...")
    for stl_path in glob.iglob(os.path.join(OUTPUT_DIR, '*.stl')):
        try:
            mesh = o3d.io.read_triangle_mesh(stl_path)
            # Sample points uniformly from the mesh surface
            point_cloud = mesh.sample_points_poisson_disk(number_of_points=100000)
            xyz_points = np.asarray(point_cloud.points, dtype=np.float32)

            xyz_filename = os.path.splitext(stl_path)[0] + '_pcd.xyz'
            np.savetxt(xyz_filename, xyz_points, fmt='%.6f', delimiter='\t')
            print(f"Saved point cloud with {len(xyz_points)} points to {xyz_filename}")
        except Exception as e:
            print(f"Failed to process {stl_path}: {e}")

    # --- 7. SURFACE RECONSTRUCTION (using external helper functions) ---
    print("\nRunning surface reconstruction and analysis...")
    input_xyz_file = os.path.join(OUTPUT_DIR, "predicted_mask_pcd.xyz")
    output_mesh_file = os.path.join(OUTPUT_DIR, "reconstructed_mesh.obj")
    
    if os.path.exists(input_xyz_file):
        # Assuming these functions are defined in 'rbf_helper_functions.py'
        process_point_cloud(input_xyz_file, output_mesh_file, color=(0.4, 0.4, 0.4), use_bpa=True)
        surface_area = compute_properties(output_mesh_file)
        
        log_file_path = os.path.join(OUTPUT_DIR, "reconstruction_log.txt")
        with open(log_file_path, "w") as log_file:
            log_file.write(f"Processed Point Cloud: {input_xyz_file}\n")
            log_file.write(f"Reconstructed Mesh: {output_mesh_file}\n")
            log_file.write(f"Surface Area: {surface_area}\n")
        print(f"Reconstruction results logged to: {log_file_path}")
        
        # visualize_mesh(output_mesh_file) # Uncomment to display mesh
    else:
        print(f"Skipping reconstruction: Input file not found at {input_xyz_file}")
        
    print("\nPipeline finished successfully.")

if __name__ == "__main__":
    main()
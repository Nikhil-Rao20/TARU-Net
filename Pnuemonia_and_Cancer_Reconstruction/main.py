# !pip install kagglehub nilearn niwidgets numpy-stl open3d k3d ipympl niwidgets --no-deps trimesh -q
print('Install the above packages if you are running this code in a Jupyter notebook or Local Machine for the first time.')

from Pnuemonia_and_Cancer_Reconstruction.Models.unet_model import UNet1
from rbf_helper_functions import *

import os, glob
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import cv2
import numpy as np
import trimesh
from skimage import morphology
from scipy import ndimage
import numpy as np
import nibabel as nib
import open3d as o3d
from sklearn.preprocessing import MinMaxScaler
import os
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from tensorflow import keras
import shutil, pathlib, fnmatch
import PIL
from niwidgets import NiftiWidget
import numpy as np
import open3d as o3d
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from skimage import measure
from stl import mesh
from sklearn.metrics import confusion_matrix
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.ndimage import binary_closing, binary_dilation, binary_erosion, generate_binary_structure

from tqdm.auto import tqdm
import torch.nn.functional as F
from torchvision.transforms.v2 import GaussianNoise
# from torchmetrics import JaccardIndex, Precision, Recall, Specificity, F1Score, AUROC
import torch.optim as optim
import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet1().to('cuda')
model.load_state_dict(torch.load('/content/U-Net_best_val_loss.pt'))


print("KaggleHub installed successfully.")
import kagglehub

# Download latest version
path = kagglehub.dataset_download("luumsk/medical-segmentation-decathlon-lung")

print("Path to dataset files:", path)


import os
import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

# Define paths
dataInputPath = '/root/.cache/kagglehub/datasets/luumsk/medical-segmentation-decathlon-lung/versions/7'
imagePathInput = os.path.join(dataInputPath, 'imagesTr/')
maskPathInput = os.path.join(dataInputPath, 'labelsTr/')

targetImageFile = 'lung_003.nii'
targetMaskFile = 'lung_003.nii'

targetImagePath = os.path.join(imagePathInput, targetImageFile)
targetMaskPath = os.path.join(maskPathInput, targetMaskFile)

def preprocess_image_and_mask(img_path, mask_path):
    # Load NIfTI files
    img_nii = nib.load(img_path).get_fdata().astype(np.float32)
    lbl_nii = nib.load(mask_path).get_fdata().astype(np.int64)

    # Ensure shape match
    if img_nii.shape != lbl_nii.shape:
        raise ValueError("Image and mask shapes do not match.")

    images_list = []
    mask_list= []

    image_data = np.zeros((256, 256, img_nii.shape[2]))

    mask_data= np.zeros((256, 256, img_nii.shape[2]))

    for i in range(img_nii.shape[2]):
        img_slice = img_nii[:, :, i]
        lbl_slice = lbl_nii[:, :, i]
        # if np.sum(lbl_slice) == 0:
        #     continue

        # Normalize image to [0,1]
        img_slice = (img_slice - img_slice.min()) / (img_slice.max() - img_slice.min() + 1e-8)

        # Convert to tensor and add channel dim
        img_tensor = torch.from_numpy(img_slice).unsqueeze(0)  # (1, H, W)
        mask_tensor = torch.from_numpy(lbl_slice).unsqueeze(0).float()  # (1, H, W)

        # Resize to (256, 256)
        img_tensor = F.interpolate(img_tensor.unsqueeze(0), size=(256, 256), mode='bilinear', align_corners=False).squeeze(0)
        mask_tensor = F.interpolate(mask_tensor.unsqueeze(0), size=(256, 256), mode='nearest').squeeze(0)
        images_list.append(img_tensor)
        mask_list.append(mask_tensor)
        image_data[:,:,i] = img_tensor
        mask_data[:,:,i] = mask_tensor
        # all_slices.append((img_tensor, mask_tensor))

    return image_data, mask_data

# prompt: calculate the dice score of predImg and imgMask

def calculate_dice_score(predImg, imgMask):

    # Ensure both inputs are binary (0 or 1)
    predImg_binary = (predImg > 0.5).astype(np.float32)
    imgMask_binary = (imgMask > 0.5).astype(np.float32)

    intersection = np.sum(predImg_binary * imgMask_binary)
    sum_masks = np.sum(predImg_binary) + np.sum(imgMask_binary)

    # Avoid division by zero
    if sum_masks == 0:
        return 1.0  # Or 0.0, depending on how you define Dice for empty masks

    dice = (2. * intersection) / sum_masks
    return dice
HOUNSFIELD_MIN = -1000
HOUNSFIELD_MAX = 2000
HOUNSFIELD_RANGE = HOUNSFIELD_MAX - HOUNSFIELD_MIN


#normalization. All constants have been defined above
def normalizeImageIntensityRange(img):
    img[img < HOUNSFIELD_MIN] = HOUNSFIELD_MIN
    img[img > HOUNSFIELD_MAX] = HOUNSFIELD_MAX
    return (img - HOUNSFIELD_MIN)/HOUNSFIELD_RANGE
#Linear interpolation to target width & height
def scaleImg(img, height, width):
    return cv2.resize(img, dsize=(width, height), interpolation=cv2.INTER_LINEAR)

def dataToMesh(vert, faces):
    stl_mesh = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, f in enumerate(faces):
        for j in range(3):
            stl_mesh.vectors[i][j] = vert[f[j], :]
    return stl_mesh

# prompt: wrte the code to plot the predImg and imgMask in 3d

def plot_3d_volume(volume, title):
    """Plots a 3D volume using marching cubes."""
    # Generate vertices and faces using marching cubes
    try:
        verts, faces, _, _ = measure.marching_cubes(volume, level=0.5)
    except ValueError as e:
        print(f"Could not perform marching cubes for {title}: {e}")
        print("This might happen if the volume is empty or constant.")
        return

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the surface
    mesh = Poly3DCollection(verts[faces])
    mesh.set_edgecolor('k')
    ax.add_collection3d(mesh)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title)

    # Auto scale axes to match the volume size
    ax.set_xlim(0, volume.shape[0])
    ax.set_ylim(0, volume.shape[1])
    ax.set_zlim(0, volume.shape[2])
    plt.savefig(title+'.png')
    print('Image Saved')



sliceIndex = 100

IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256

SLICE_X = True
SLICE_Y = True
SLICE_Z = True


# Run preprocessing
imgTarget, imgMask = preprocess_image_and_mask(targetImagePath, targetMaskPath)


predictions = np.zeros((256, 256, 288))

with torch.no_grad():
    for idx in range(imgTarget.shape[-1]):
        img1 = imgTarget[:, :, idx][np.newaxis, np.newaxis, :, :]
        img_tensor = torch.from_numpy(img1).float().cuda()
        out = model(img_tensor)
        prediction_slice = out.squeeze().cpu()
        predictions[:, :, idx] = (torch.sigmoid(prediction_slice).numpy()>0.5).astype(np.int64)

# Calculate and print the Dice score
dice_score = calculate_dice_score(predictions, imgMask)
print(f"Dice Score: {dice_score}")



from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from stl import mesh


predImg = predictions

# Generate vertices and faces using marching cubes
vertices, faces, _, _ = measure.marching_cubes(predImg, level=0.5)
ctvertices, ctfaces, _, _ = measure.marching_cubes(imgTarget, level=0.5)
maskvertices, maskfaces, _, _ = measure.marching_cubes(imgMask, level=0.5)

# Optional: Display the number of vertices and faces for debugging
print(f"Inferred Mask: {len(vertices)} vertices, {len(faces)} faces")
print(f"Input Scan: {len(ctvertices)} vertices, {len(ctfaces)} faces")
print(f"Annotated Mask: {len(maskvertices)} vertices, {len(maskfaces)} faces")

plot_3d_volume(predImg, "PredCancer")
plot_3d_volume(imgMask, "GroundCancer")



output_path = './'  # Define your output directory
inference_mesh = dataToMesh(vertices, faces)
inference_mesh.save(output_path + 'Inferenced_lung_003.stl')

mask_mesh = dataToMesh(maskvertices, maskfaces)
mask_mesh.save(output_path + 'Mask_lung_003.stl')

print("STL files saved successfully.")


stl_file_path = "/content/Inferenced_lung_003.stl"  # Adjust path if needed
# Read the .stl file using Open3D
mesh = o3d.io.read_triangle_mesh(stl_file_path)
print("_")
# Sample points using Poisson disk sampling
pointcloud = mesh.sample_points_poisson_disk(100000)
# Convert sampled points to a numpy array
xyz_load = np.asarray(pointcloud.points, dtype=np.float32)
print('xyz_load shape', xyz_load.shape)

for filename in sorted(glob.iglob("./" + '*.stl')):
    mesh = o3d.io.read_triangle_mesh(filename)
    print("Mesh read complete")
    pointcloud = mesh.sample_points_poisson_disk(100000)
    print("PCD generation complete")
    xyz_load = np.asarray(pointcloud.points,dtype=np.float32)
    print('xyz_load shape', xyz_load.shape)
    with open('{}_pcd.xyz'.format(filename[:-4]), 'w') as f:
        for line in xyz_load:
            f.write("{}\t {}\t {}\t".format(str(line[0]),str(line[1]),str(line[2])))
            f.write('\n')


input_xyz_file = "/content/Inferenced_lung_010_pcd.xyz"
output_dir = "/content/results"
os.makedirs(output_dir, exist_ok=True)
output_mesh_file = os.path.join(output_dir, "Inferenced_lung_003_mesh.obj")


process_point_cloud(input_xyz_file, output_mesh_file, color=(0.4, 0.4, 0.4), use_bpa=True)

surface_area = compute_properties(output_mesh_file)
log_file_path = os.path.join(output_dir, "results_log.txt")
with open(log_file_path, "w") as log_file:
    log_file.write(f"Processed Mesh: {output_mesh_file}\n")
    log_file.write(f"Surface Area: {surface_area}\n")

print(f"Results logged in: {log_file_path}")

visualize_mesh(output_mesh_file)



import numpy as np
import nibabel as nib
import open3d as o3d
import os # Import the os module

ct = nib.load('/root/.cache/kagglehub/datasets/andrewmvd/covid19-ct-scans/versions/4/ct_scans/coronacases_org_001.nii').get_fdata()
mask = nib.load('/root/.cache/kagglehub/datasets/andrewmvd/covid19-ct-scans/versions/4/infection_mask/coronacases_001.nii').get_fdata()
lung_mask = mask > 0


lung_indices = np.argwhere(lung_mask)
zmin, ymin, xmin = lung_indices.min(axis=0)
zmax, ymax, xmax = lung_indices.max(axis=0)

print(f"Lung bbox: Z({zmin}-{zmax}), Y({ymin}-{ymax}), X({xmin}-{xmax})")

lung_bbox_min = np.array([xmin, ymin, zmin])
lung_bbox_max = np.array([xmax, ymax, zmax])

stl_file_path = '/content/Mask_lung_010.stl'

# Check if the file exists before attempting to load
if not os.path.exists(stl_file_path):
    print(f"Error: STL file not found at {stl_file_path}")
else:
    mesh = o3d.io.read_triangle_mesh(stl_file_path)
    vertices = np.asarray(mesh.vertices)

    if vertices.shape[0] == 0:
        print(f"Error: No vertices loaded from {stl_file_path}")
    else:
        vmin = vertices.min(axis=0)
        vmax = vertices.max(axis=0)
        vertices_normalized = (vertices - vmin) / (vmax - vmin)


        lung_bbox_size = lung_bbox_max - lung_bbox_min
        vertices_voxel = vertices_normalized * lung_bbox_size[::-1] + lung_bbox_min[::-1]
        ct_shape = ct.shape
        vertices_voxel = np.clip(vertices_voxel, 0, np.array(ct_shape[::-1]) - 1).astype(int)


        colors = []

        for idx in range(vertices_voxel.shape[0]):
            z, y, x = vertices_voxel[idx]

            if not (0 <= z < ct_shape[0] and 0 <= y < ct_shape[1] and 0 <= x < ct_shape[2]):
                colors.append([0.5, 0.5, 0.5])
                continue

            if lung_mask[z, y, x] == 0:
                colors.append([0.5, 0.5, 0.5])
                continue

            hu = ct[z, y, x]

            # New thresholds
            if hu < -800:
                colors.append([0, 1, 0])  # Green (healthy)
            elif -800 <= hu <= -400:
                colors.append([1, 1, 0])  # Yellow (moderate infection)
            else:
                colors.append([1, 0, 0])  # Red (severe infection)


        colors = np.array(colors)
        mesh.vertex_colors = o3d.utility.Vector3dVector(colors)


        o3d.io.write_triangle_mesh('/content/colored_lung_True_attempt.obj', mesh)

        print("Done making the mesh")


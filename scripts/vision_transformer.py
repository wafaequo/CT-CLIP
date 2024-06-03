# -*- coding: utf-8 -*-
"""
Created on Wed May 29 16:22:33 2024

@author: 20192032
"""
#%%CONVERTING TO NII.
#after preprocessing, all images are saved as tensors. Convert back to images to inspect + use as input.  
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import torch
import os


def tensor_to_nifti(tensor, path, affine=np.eye(4)):
    """
    Save tensor as a NIfTI file.

    Args:
        tensor (torch.Tensor): The input tensor with shape (D, H, W) or (C, D, H, W).
        path (str): The path to save the NIfTI file.
        affine (np.ndarray, optional): The affine matrix for the NIfTI file. Defaults to np.eye(4).
    """

    tensor = tensor.cpu()

    if tensor.dim() == 4:
        # Assume single channel data if there are multiple channels
        if tensor.size(0) != 1:
            print("Warning: Saving only the first channel of the input tensor")
        tensor = tensor.squeeze(0)
    tensor=tensor.swapaxes(0,2)
    numpy_data = tensor.detach().numpy().astype(np.float32)
    nifti_img = nib.Nifti1Image(numpy_data, affine)
    nib.save(nifti_img, path)
    


#saving npz as nii images
preprocessed_dir = r"D:\dataset\valid_preprocessed"
images_dir = r"D:\dataset\valid_images"

for i in range(1, 80):
    for suffix in ['a', 'b']:
        for file_num in [1, 2]:
            npz_file = os.path.join(preprocessed_dir, f"valid_{i}", f"valid_{i}{suffix}", f"valid_{i}_{suffix}_{file_num}.npz")
            if os.path.exists(npz_file):
                npz_data = np.load(npz_file)
                numpy_data = npz_data["arr_0"]
                tensor_data = torch.tensor(numpy_data)
                
                save_dir = os.path.join(images_dir, f"valid_{i}", f"valid_{i}_{suffix}")
                
                os.makedirs(save_dir, exist_ok=True)
                
                nifti_file = os.path.join(save_dir, f"valid_{i}_{suffix}_{file_num}.nii")
                
                tensor_to_nifti(tensor_data, nifti_file)
                
                print(f"Converted and saved image {i} with suffix '{suffix}' and file number {file_num}")
            else:
                print(f"File not found for image {i} with suffix '{suffix}' and file number {file_num}")
    
            
#%%VISUALISE IMAGE TO COMPARE PREPROCESSED WITH ORIGINAL
import nibabel as nib
import matplotlib.pyplot as plt

original_image = r"D:\dataset\valid\valid_1\valid_1_a\valid_1_a_1.nii.gz"
pp_image = r"D:\dataset\valid_images\valid_1\valid_1_a\valid_1_a_1.nii"


img_original = nib.load(original_image)
img_pp = nib.load(pp_image)


data_original = img_original.get_fdata()
data_pp = img_pp.get_fdata()

#original image visualization
num_slices = data_original.shape[-1]  # Number of slices along the last dimension
middle_slice = num_slices // 2  # Choose the middle slice
plt.imshow(data_original[:, :, middle_slice], cmap='gray')
plt.axis('off')
plt.title('Middle slice')
plt.show()

#pp image
num_slices = data_pp.shape[-1]  # Number of slices along the last dimension
middle_slice = num_slices // 2  # Choose the middle slice
plt.imshow(data_pp[:, :, middle_slice], cmap='gray')
plt.axis('off')
plt.title('Middle slice')
plt.show()

#%%LOADING PRETRAINED CTCLIP MODEL
import torch
from transformer_maskgit import CTViT
from transformers import BertTokenizer, BertModel
from ct_clip import CTCLIP
from zero_shot import CTClipInference
import accelerate

image_encoder = CTViT(
    dim = 512,
    codebook_size = 8192,
    image_size = 480,
    patch_size = 30,
    temporal_patch_size = 15,
    spatial_depth = 4,
    temporal_depth = 4,
    dim_head = 32,
    heads = 8
)

pretrained_weights_path = 'CT_CLIP_zeroshot.pt'
state_dict = torch.load(pretrained_weights_path, map_location=torch.device('cpu'))
filtered_state_dict = {key.replace('visual_transformer.', ''): value for key, value in state_dict.items() if key.startswith('visual_transformer.')}
image_encoder.load_state_dict(filtered_state_dict)
#%%INFERENCE
from transformer_maskgit.ctvit_inference import CTVIT_inf

inference_engine = CTVIT_inf(
    vae=image_encoder,  
    batch_size=1,  
    folder=r'D:\dataset\images2',  # Path to the folder containing your image data
    train_on_images=True,  #False for videos
    results_folder='./results',
    valid_frac=0.3,  # Fraction of data to use for validation
    random_split_seed=42,  # Seed for random split
    use_ema=True,
    num_train_steps=1
)

inference_engine.infer()

#%%FEATURE EXTRACTING
import nibabel as nib
from torchvision import transforms as T
import numpy as np
import torch
from pathlib import Path
import os
#from skimage.transform import resize


image_encoder.eval()

folder = r'D:\dataset\pp_arrays'

#aanname dat dit niet wordt gedaan gezien .nii format.
# preprocess = transforms.Compose([
#     T.Resize(image_encoder.image_size),
#     T.RandomHorizontalFlip(),
#     T.Tensor()
# ])

# Define the list of file extensions
exts = ['npz']
def load_and_preprocess_npz(file_path):
    img_array = np.load(file_path)
    resized_array = np.resize(img_array['arr_0'], image_encoder.image_size)

    # Convert to PyTorch tensor
    img_tensor = torch.Tensor(resized_array)
    return img_tensor

# Load and preprocess multiple images from the folder
image_paths = [p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')]
tensor_list = [load_and_preprocess_npz(file_path) for file_path in image_paths]

batch_images = torch.stack(tensor_list)
batch_images = batch_images.unsqueeze(0)

# Forward pass
with torch.no_grad():
    output = image_encoder(batch_images)

#%%
image_encoder.eval()
features = {}
for path in image_paths:
    img_array = np.load(path)
    resized_array = np.resize(img_array['arr_0'], image_encoder.image_size)
    img_tensor = torch.Tensor(resized_array)
    with torch.no_grad():
        feature = image_encoder(img_tensor)
    features[path] = feature
    
print(features)
























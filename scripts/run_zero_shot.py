#%% IMPORT STATEMENTS
import torch
from transformer_maskgit import CTViT
from transformers import BertTokenizer, BertModel
from ct_clip import CTCLIP
from zero_shot import CTClipInference
import accelerate
#%%
tokenizer = BertTokenizer.from_pretrained('microsoft/BiomedVLP-CXR-BERT-specialized',do_lower_case=True)
text_encoder = BertModel.from_pretrained("microsoft/BiomedVLP-CXR-BERT-specialized")

text_encoder.resize_token_embeddings(len(tokenizer))

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

clip = CTCLIP(
    image_encoder = image_encoder,
    text_encoder = text_encoder,
    dim_image = 2097152,
    dim_text = 768,
    dim_latent = 512,
    extra_latent_projection = False,         # whether to use separate projections for text-to-image vs image-to-text comparisons (CLOOB)
    use_mlm=False,
    downsample_image_embeds = False,
    use_all_token_embeds = False

)

# Assuming you have the full path to the model file
full_path = r"C:\Users\20192032\OneDrive - TU Eindhoven\Documents\GitHub\CT-CLIP2\scripts\CT_CLIP_zeroshot.pt"

# Load the model using the full path
clip.load(full_path)

#%% RUN INFERENCE
image_path = r"D:\dataset\valid"
label_path = r"C:\Users\20192032\OneDrive - TU Eindhoven\Documents\GitHub\CT-CLIP2\scripts\valid_label.csv"
result_path = r"C:\Users\20192032\OneDrive - TU Eindhoven\Documents\GitHub\CT-CLIP2\scripts\results"
inference = CTClipInference(
    CTClip=clip,
    data_folder=image_path,  # Path to the folder containing images
    labels=label_path,  # If ground truth labels are available
    batch_size=4,  # Adjust batch size based on your system's memory
    results_folder=result_path,
    num_train_steps=1000,  # Adjust based on your actual training steps
)
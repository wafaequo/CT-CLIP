import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.args import parse_arguments
from transformers import BertTokenizer, BertModel
from transformer_maskgit import CTViT
from ct_clip import CTCLIP
from data_inference import CTReportDatasetinfer
from eval import evaluate_internal, plot_roc, accuracy, sigmoid, bootstrap, compute_cis
import tqdm
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, multilabel_confusion_matrix, f1_score, accuracy_score
import os
import copy

def sigmoid(tensor):
    return 1 / (1 + torch.exp(-tensor))

class ImageLatentsClassifier(nn.Module):
    def __init__(self, trained_model, latent_dim, dropout_prob=0.3):
        super(ImageLatentsClassifier, self).__init__()
        self.trained_model = trained_model
        self.dropout = nn.Dropout(dropout_prob)  # Add dropout layer
        self.relu = nn.ReLU()

    def forward(self, latents=False, *args, **kwargs):
        kwargs['return_latents'] = True
        _, image_latents = self.trained_model(*args, **kwargs)
        image_latents = self.relu(image_latents)
        image_latents = self.dropout(image_latents)
        return image_latents


    def save(self, file_path):
        torch.save(self.state_dict(), file_path)
        
    def load(self, file_path):
        
        loaded_state_dict = torch.load(file_path, map_location=torch.device('cpu'))
        model_state_dict = self.state_dict()
        
        filtered_state_dict = {k: v for k, v in loaded_state_dict.items() if k in model_state_dict and model_state_dict[k].shape == v.shape}
        
        model_state_dict.update(filtered_state_dict)
        self.load_state_dict(model_state_dict)

def evaluate_model(args, model, dataloader, device):
    model = model.to(device)
    model.eval()  # Set the model to evaluation mode
    all_features = []
    accs = []
    with torch.no_grad():

        for batch in tqdm.tqdm(dataloader):
            inputs, acc_no = batch
            inputs = inputs.to(device)
            features = model(inputs, device=device)
            all_features.append(features.cpu().numpy())
            accs.append(acc_no[0])
            print(acc_no[0], flush=True)
            
        plotdir = args.save
        os.makedirs(plotdir, exist_ok=True)
        
        all_features = np.concatenate(all_features, axis=0)

        with open(f"{plotdir}accessions.txt", "w") as file:
            for item in accs:
                file.write(item[0] + "\n")
                
        np.savez(f"{plotdir}features.npz", data=all_features)

if __name__ == '__main__':
    args = parse_arguments()  # Assuming this function provides necessary arguments


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
            text_encoder = None,
            dim_image = 2097152,
            dim_text = 0,
            dim_latent = 512,
            extra_latent_projection = False,         # whether to use separate projections for text-to-image vs image-to-text comparisons (CLOOB)
            use_mlm=False,
            downsample_image_embeds = False,
            use_all_token_embeds = False

        )
    
    image_classifier = ImageLatentsClassifier(clip, 512)
    zero_shot = copy.deepcopy(image_classifier)

    image_classifier.load(args.pretrained)  # Assuming args.checkpoint_path is the path to the saved checkpoint


    # Prepare the evaluation dataset
    ds = CTReportDatasetinfer(data_folder=args.data_folder, min_slices=20, resize_dim=500)
    dl = DataLoader(ds, num_workers=4, batch_size=12, shuffle=False)

    # Evaluate the model
    evaluate_model(args, image_classifier, dl, torch.device('cpu'))
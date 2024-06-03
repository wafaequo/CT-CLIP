# -*- coding: utf-8 -*-
"""
Created on Sun Jun  2 18:36:32 2024

@author: 20192032
"""
import subprocess

# Define the command and arguments
command = [
    'python', 'ct_lipro_inference.py',
    '--lr', '1e-5',
    '--wd', '0.1',
    '--epochs', '10',
    '--warmup_length', '10000',
    '--save', r'C:\Users\20192032\OneDrive - TU Eindhoven\Documents\GitHub\CT-CLIP2\scripts',
    '--pretrained', r'C:\Users\20192032\OneDrive - TU Eindhoven\Documents\GitHub\CT-CLIP2\scripts\CT_LiPro.pt',
    '--data-folder', r'D:\dataset\valid_preprocessed',
    '--labels', r'D:\dataset\valid_labels.csv'
]

# Run the command
result = subprocess.run(command, capture_output=True, text=True)

# Print the output and errors, if any
print("Output:")
print(result.stdout)
print("Errors:")
print(result.stderr)

#%%
import os
current_directory = os.getcwd()
print("Current Working Directory:", current_directory)


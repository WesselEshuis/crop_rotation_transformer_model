import pandas as pd
import numpy as np
import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from Transformer_model_variable import CropRotationTransformer
from Crop_encoder_list import crop_type_individual_encoding

# This script returns a pdf containing the distribution of predicted probabilities from the crop rotation transformer
# Output file name:
save_file = 'model_prediction.pdf'
# It requires the user to enter a comma seperated sequence, with a length of minimum 1 to maximum 7 crops

# Enter the sequence below, an example sequence of 4 crops is given (WinterWheat>Maize>Potatoes>Maize)
sequence_test = [22, 6, 11, 6]
# The crops are represented by numbers, the encoding is listed in the Crop_encoder_list.py

# Ensuring the correct folder is selected
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# Ensure you're loading the model from the correct file path
transformer_model_path = "crop_rotation_model.pth"

# Reverse the dictionary to map indexes to crop names
index_to_crop = {}
for crop, index in crop_type_individual_encoding.items():
    if index not in index_to_crop:
        index_to_crop[index] = []
    index_to_crop[index].append(crop)

# Function to map indexes to crop names
def map_indexes_to_crops(indexes):
    crop_names = []
    for index in indexes:
        crops = index_to_crop.get(index, ["Unknown crop"])
        crop_names.append(crops)
    return crop_names

def compute_TD(TD_input_sequence, transformer_model_path):
    # Load the model
    device = torch.device('cpu')
    model = CropRotationTransformer(
        num_crops=26,                       # Number of main crop classes
        num_embeddings=27,                  # Total number of embeddings
        embedding_dim=128,                  # Embedding dimension
        num_heads=4,                        # Number of attention heads
        num_layers=2,                       # Number of transformer layers
        dropout=0.1,                        # Dropout rate
        device=device                       # Device (CPU/GPU)
    ).to(device)

    # Load model's state dict
    model.load_state_dict(torch.load(transformer_model_path, map_location='cpu'))
    model.eval().to(device)
        
    main_sequence = torch.tensor(TD_input_sequence).to(device)  # Main crop sequence
    cover_sequence = torch.tensor([0]).to(device)  # Cover crop sequence

    # Add batch dimension
    main_sequence = main_sequence.unsqueeze(0)  # Shape: [1, 1]
    cover_sequence = cover_sequence.unsqueeze(0)  # Shape: [1, 2]

    # Make predictions
    with torch.no_grad():
        outputs = model(main_sequence, cover_sequence)  # Adjust if your model takes other inputs
        main_crop_output, _ = outputs  # Unpack outputs
        Td = F.softmax(main_crop_output, dim=1).cpu().numpy().flatten()
    return Td

def Main(sequence, transformer_model_path):
    
    # Preprocessing for TD
    TD_input_sequence = sequence
    Td = compute_TD(TD_input_sequence, transformer_model_path)

    return Td

# Call the Main function with the sequence and other parameters

Td = Main(sequence_test, transformer_model_path=transformer_model_path)

# Convert probabilities to percentages
Td = np.array(Td) * 100

# Convert lists of names to single strings (handling multiple names per crop index)
crop_labels = ["/".join(index_to_crop[i]) if i in index_to_crop else f"Crop {i}" for i in range(26)]

# Create DataFrame
df = pd.DataFrame({'Td': Td}, index=crop_labels)

# Sort by Transformer prediction (Td)
df = df.sort_values(by='Td', ascending=False)

# Define custom colors (adjust as needed)
colors = {  
    'Td': '#1f77b4'   
}

# Set up bar width and positions
x = np.arange(len(df))  # X positions for crops
width = 0.5  # Bar width

# Create figure
fig, ax = plt.subplots(figsize=(12, 6))

ax.bar(x + 2*width, df['Td'], width, label='TD', color=colors['Td'])

# Labels and legend
ax.set_xlabel('Crop Type')
ax.set_ylabel('Probability (%)')  # Now in percentages
ax.set_title('Crop Probability Distributions for the sequence: Sugar beets - Winter wheat - Potatoes - Winter wheat ')
ax.set_xticks(x)
ax.set_xticklabels(df.index, rotation=45, ha='right')
ax.legend()

# Save the chart to a PDF
save_path = os.path.join(script_dir, save_file)

plt.tight_layout()

# Example: Save some data to that file
with open(save_path, "w") as file:
    file.write("This file is saved in the same folder as model.pth!")

plt.savefig(save_path)  # Save the plot
plt.close()

import json
import os
import cv2
import numpy as np
import torch
from tqdm import tqdm
from plami_inference import PLAMi
# 预设参数
model_path = './checkpoints/plami_v1.5_7b'
data_folder = '/root/autodl-tmp/PLAMi/data/imgs'  
json_file = '/root/autodl-tmp/PLAMi/data/PLAMi-112K/filtered_plami_detail_description.json'
description_type = 'short description'  # 或 'detailed description'

# 初始化 PLAMi 模型
device = 'cuda' if torch.cuda.is_available() else 'cpu'
plami_model = PLAMi(model_path=model_path, device=device)

def annToMask(mask_ann, h, w):
    """ Convert annotation polygons to a binary mask """
    mask = np.zeros((h, w), dtype=np.uint8)
    if isinstance(mask_ann, list):
        for shape in mask_ann:
            if isinstance(shape, list):
                points = np.array([shape], dtype=np.int32)
                points = points.reshape((-1, 1, 2))
                cv2.fillPoly(mask, [points], 1)
    else:
        raise ValueError("mask_ann needs to be a list of polygons")
    return mask

def load_image(file_path):
    return cv2.imread(file_path)

def process_images(json_path, image_dir):
    # Load JSON file
    with open(json_path, 'r') as f:
        data = json.load(f)

    results = []

    # Process each image
    for item in tqdm(data, desc="Processing images"):
        image_path = os.path.join(image_dir, item['file_name'])
        img = load_image(image_path)
        if img is None:
            print(f"Failed to load {item['file_name']}, skipping...")
            continue

        # Convert JSON polygon annotations to binary masks and predict descriptions
        h, w = item['height'], item['width']
        predicted_descriptions = []
        for ann, orig_desc in zip(item['annotation'], item['description']):
            mask = annToMask(ann['segmentation'], h, w)
            description = plami_model.plami_predict(img, mask, type=description_type)
            predicted_descriptions.append(description)
            print(f"Processed {item['id']} - {description}")

        # Save results
        results.append({
            'id': item['id'],
            'file_name': item['file_name'],
            'original_description': item['description'],
            'predicted_description': predicted_descriptions
        })

    # Save results to a new JSON file
    output_json_path = '/root/autodl-tmp/PLAMi/demo/results/processed_descriptions.json'
    if not os.path.exists(os.path.dirname(output_json_path)):
        os.makedirs(os.path.dirname(output_json_path))
    with open(output_json_path, 'w') as f:
        json.dump(results, f, indent=4)

    print("All processing completed. Results saved to:", output_json_path)

    return results

# Execute the processing
process_images(json_file, data_folder)

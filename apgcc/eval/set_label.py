import os
import json
import pandas as pd
import glob
from assets.config import LABEL_EXTRACT_DIR

def setting_extract_label():
    os.makedirs('label', exist_ok=True)
    json_folders = [folder for folder in os.listdir(LABEL_EXTRACT_DIR) if folder.endswith("_json")]

    for folder in json_folders:
        data = []
        
        folder_path = os.path.join(LABEL_EXTRACT_DIR, folder)
        
        json_files = glob.glob(os.path.join(folder_path, "*.json"))
        
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8-sig') as f:
                    json_data = json.load(f)
                
                # Extract imagename and counting
                image_name = json_data.get('image', {}).get('imagename', '')
                counting = json_data.get('image', {}).get('crowdinfo', {}).get('counting', 0)
                
                data.append({
                    'image_name': image_name,
                    'crowd_counting': counting
                })
            except Exception as e:
                print(f"Error processing {json_file}: {e}")
        
        df = pd.DataFrame(data)
        csv_filename = folder.replace("_json", "") + ".csv"
        output_path = os.path.join('label', csv_filename)
        
        df.to_csv(output_path, index=False)
        
        print(f"Processed {len(data)} files from {folder} and saved to {output_path}")

    print("All JSON folders have been processed and CSV files created in the 'label' directory.")



    
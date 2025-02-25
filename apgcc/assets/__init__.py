from huggingface_hub import hf_hub_download
import os

def download_required_files():
    """Initialize required files from Hugging Face Hub"""
    try:
        cache_dir = "assets/"
        if not os.path.exists(os.path.join(cache_dir, "CLIP_EBC_nwpu_rmse.pth")):
            hf_hub_download(
                repo_id="PIA-SPACE-LAB/PIA_KTT_PersonDet_v3.1.3",
                filename="PersonDet_v3.1.3.pt",
                # cache_dir=cache_dir,
                local_dir=cache_dir
            )
        print("Required files downloaded successfully")
    except Exception as e:
        print(f"Error downloading required files: {e}")

download_required_files()
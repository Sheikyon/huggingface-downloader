import os
from pathlib import Path
from rich import print
from huggingface_hub import hf_hub_download, model_info

def huggingface_downloader():
    model_repo = input("Enter the Hugging Face model repo (e.g., deepseek-ai/DeepSeek-V3.2): ").strip()

    base_dir = Path(__file__).resolve().parents[2] / "models"
    os.makedirs(base_dir, exist_ok=True)

    model_name = model_repo.split("/")[-1]
    model_dir = base_dir / model_name
    os.makedirs(model_dir, exist_ok=True)

    print(f"Downloading model '{model_repo}' to '{model_dir}' ...")

    for file in model_info(model_repo).siblings:
        try:
            hf_hub_download(
                repo_id=model_repo,
                filename=file.rfilename,
                cache_dir=model_dir,
                local_files_only=False
            )
            print(f"Downloaded: {file.rfilename}")
        except Exception as e:
            print(f"Failed to download {file.rfilename}: {e}")

    print(f"Model '{model_repo}' downloaded successfully to {model_dir}")
    return model_dir

huggingface_downloader()
import os
from pathlib import Path
from rich import print
from huggingface_hub import snapshot_download
from huggingface_hub.utils import RepositoryNotFoundError

def huggingface_downloader():
    model_repo = input("Enter the Hugging Face model repo (e.g., deepseek-ai/DeepSeek-V3.2): ").strip()
    if not model_repo:
        print("[red]No repo provided. Aborting.[/red]")
        return None
   
    base_dir = Path(__file__).resolve().parent / "models"
    os.makedirs(base_dir, exist_ok=True)
   
    model_name = model_repo.split("/")[-1]
    model_dir = base_dir / model_name
   
    os.makedirs(model_dir, exist_ok=True)
   
    print(f"Downloading model '{model_repo}' to '{model_dir}' ...")
   
    try:
        snapshot_download(
            repo_id=model_repo,
            local_dir=str(model_dir),
            local_dir_use_symlinks=False, 
            revision="main"
        )
        print(f"[bold green]Model '{model_repo}' downloaded successfully to {model_dir}[/bold green]")
    except RepositoryNotFoundError:
        print(f"[red]Model repo '{model_repo}' does not exist or is private.[/red]")
        return None
    except Exception as e:
        print(f"[red]Failed to download '{model_repo}': {e}[/red]")
        return None
   
    return model_dir

huggingface_downloader()

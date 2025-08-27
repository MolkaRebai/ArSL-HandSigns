import kagglehub
import zipfile
import os
import warnings

# Suppress version warning
warnings.filterwarnings("ignore", category=UserWarning, message=".*outdated kagglehub version.*")


def download_dataset():
    """Downloads and extracts the dataset if not already present"""
    try:
        print("Checking dataset...")
        dataset_path = kagglehub.dataset_download(
            "muhammadalbrham/rgb-arabic-alphabets-sign-language-dataset"
        )

        dataset_folder = "dataset"

        # Skip if dataset already exists
        if os.path.exists(dataset_folder) and os.listdir(dataset_folder):
            print("Dataset already exists. Skipping download.")
            return dataset_folder

        # Handle ZIP download
        if dataset_path.endswith(".zip"):
            os.makedirs(dataset_folder, exist_ok=True)
            with zipfile.ZipFile(dataset_path, 'r') as zip_ref:
                zip_ref.extractall(dataset_folder)
            print(f"Dataset extracted to: {dataset_folder}")
            return dataset_folder

        return dataset_path

    except Exception as e:
        print(f"Error downloading dataset: {str(e)}")
        return None


if __name__ == "__main__":
    download_dataset()
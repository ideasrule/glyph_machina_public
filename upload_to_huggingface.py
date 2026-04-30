"""Upload train_list.csv / test_list.csv (with line images) to the HuggingFace Hub.

Each CSV row references an image at ``file_name`` (relative to this script's
directory). We build a ``datasets.Dataset`` whose ``image`` column is a
``datasets.Image`` feature so the actual PNG bytes are bundled into the dataset
shards on the Hub instead of left as dangling paths.
"""
import argparse
from pathlib import Path

import pandas as pd
from datasets import Dataset, DatasetDict, Features, Image, Value
from huggingface_hub import HfApi, login

REPO_ID = "mzzhang2014/glyph_machina"
ROOT = Path(__file__).resolve().parent


def build_split(csv_path: Path) -> Dataset:
    df = pd.read_csv(csv_path)
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")])

    df["image"] = df["file_name"].map(lambda p: str(ROOT / p))

    missing = [p for p in df["image"] if not Path(p).is_file()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} image(s) referenced by {csv_path.name} are missing, "
            f"e.g. {missing[0]}"
        )

    features = Features(
        {
            "file_name": Value("string"),
            "xml_filename": Value("string"),
            "text": Value("string"),
            "image": Image(),
        }
    )
    return Dataset.from_pandas(
        df[["file_name", "xml_filename", "text", "image"]],
        features=features,
        preserve_index=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default=REPO_ID)
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()

    if not HfApi().whoami():
        login()

    splits = DatasetDict(
        {
            "train": build_split(ROOT / "train_list.csv"),
            "test": build_split(ROOT / "test_list.csv"),
        }
    )

    print(splits)
    splits.push_to_hub(args.repo_id, private=args.private)

    # Also push the source train_data/ and test_data/ folders so the XML
    # ground-truth files and full-page JPGs are available alongside the
    # parquet shards. The line_*.png crops are already inside the parquet
    # shards via the Image feature, so we skip them here to avoid duplication.
    api = HfApi()
    for folder in ("train_data", "test_data"):
        local = ROOT / folder
        if not local.is_dir():
            print(f"skipping {folder}: directory not found")
            continue
        print(f"uploading {folder}/ (xml + jpg only)")
        api.upload_folder(
            folder_path=str(local),
            path_in_repo=folder,
            repo_id=args.repo_id,
            repo_type="dataset",
            allow_patterns=["*.xml", "*.jpg", "*.jpeg", "*.png"],
            ignore_patterns=["line_*.png"],
            delete_patterns=["**/.ipynb_checkpoints/**"]
        )

    print(f"Uploaded to https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()

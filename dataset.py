from torch.utils.data import Dataset
import pandas as pd
import torch
from PIL import Image, ImageOps

class LineImageDataset(Dataset):
    def __init__(self, line_list_filename, char_to_num, transform=None):
        self.df = pd.read_csv(line_list_filename)
        self.char_to_num = char_to_num
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_name = self.df['file_name'][idx]
        text = self.df['text'][idx]

        image = Image.open(file_name).convert("L")

        if self.transform is not None:
            image = self.transform(image)


        num_labels = torch.tensor([self.char_to_num[c] for c in text])
        return {"image": image, "target": num_labels, "text": text}

    def get_xml_filenames(self):
        return self.df["xml_filename"].unique()

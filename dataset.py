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

class LineDatasetTrOCR(Dataset):
    def __init__(self, root_dir, df, processor, max_target_length=128, transform=None):
        self.root_dir = root_dir
        self.df = df
        self.processor = processor
        self.max_target_length = max_target_length
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # get file name + text 
        file_name = self.df['file_name'][idx]
        text = self.df['text'][idx]
        # prepare image (i.e. resize + normalize)
        image = Image.open(self.root_dir + file_name).convert("RGB")
        image = ImageOps.invert(image)
       
        if self.transform is not None:
            image = self.transform(image)

        pixel_values = self.processor(image, return_tensors="pt").pixel_values
        # add labels (input_ids) by encoding the text
        labels = self.processor.tokenizer(text, 
                                          padding="max_length", 
                                          max_length=self.max_target_length).input_ids
        # important: make sure that PAD tokens are ignored by the loss function
        labels = [label if label != self.processor.tokenizer.pad_token_id else -100 for label in labels]

        encoding = {"pixel_values": pixel_values.squeeze(), "labels": torch.tensor(labels)}
        return encoding
    

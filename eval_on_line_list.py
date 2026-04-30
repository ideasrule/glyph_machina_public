import argparse
import glob
import os
import re
import xml.etree.ElementTree as ET
import pandas as pd

import sys
import numpy as np
import torch
from PIL import Image, ImageOps
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchmetrics.text import CharErrorRate, WordErrorRate
import torchvision.transforms.v2 as transforms
from model import HTRModel, CTC_DECODER
from dataset import LineImageDataset
from pyctcdecode import build_ctcdecoder
from model import NUM_TO_CHAR, CHAR_TO_NUM, IMAGE_MEAN, IMAGE_STD

val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[IMAGE_MEAN], std=[IMAGE_STD]),
])

def get_prediction(output):      
    rot_output = torch.squeeze(output).T.detach().cpu().numpy()
    rot_output = np.roll(rot_output, -1, axis=1)
    beams = CTC_DECODER.decode_beams(rot_output)
    beam_prediction = beams[0][0]
    beam_logit_score = beams[0][3] 
    beam_prediction = CTC_DECODER.decode(rot_output)
    return beam_prediction, beam_logit_score
    

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

net = HTRModel()
state = torch.load(sys.argv[1], map_location=device, weights_only=True)
net.load_state_dict(state)
net.to(device)
net.eval()

dataset = LineImageDataset(sys.argv[2], CHAR_TO_NUM, transform=val_transform)
loader = DataLoader(dataset, batch_size=1, num_workers=4)
print("Num val lines:", len(dataset))

grand_cer_calc = CharErrorRate()
grand_wer_calc = WordErrorRate()
df = dataset.df
print("XML_filename,CER,WER,num_lines")
for xml_filename in df["xml_filename"].unique():
    cer_calc = CharErrorRate()
    wer_calc = WordErrorRate()
    
    subset = df[df["xml_filename"] == xml_filename]
    for index, row in subset.iterrows():
        image = dataset[index]["image"].to(device).unsqueeze(0)
        target = dataset[index]["target"].unsqueeze(0)
        with torch.no_grad():
            output = net(image)
        prediction, _ = get_prediction(output)
        truth = row["text"]
        prediction = prediction.replace("u", "v").replace("U", "V").replace("i", "j").replace("I", "J")
        truth = truth.replace("u", "v").replace("U", "V").replace("i", "j").replace("I", "J")
        
        cer_calc.update(truth, prediction)
        wer_calc.update(truth, prediction)
        grand_cer_calc.update(truth, prediction)
        grand_wer_calc.update(truth, prediction)

    cer = cer_calc.compute().item()
    wer = wer_calc.compute().item()
    print(f"{xml_filename},{cer:.4f},{wer:.4f},{len(subset)}")

grand_cer = grand_cer_calc.compute().item()
grand_wer = grand_wer_calc.compute().item()
print(f"Combined CER: {grand_cer:.4f}, combined WER: {grand_wer:.4f}")

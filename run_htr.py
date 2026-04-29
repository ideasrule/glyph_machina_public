from torch import nn, utils
import pdb
import torch
import re
import numpy as np
import PIL
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import xml.etree.ElementTree as ET
import os.path
import sys
from pyctcdecode import build_ctcdecoder
import time
import matplotlib.pyplot as plt
from model import HTRModel, ALL_CHARS, NUM_TO_CHAR, CTC_DECODER, IMAGE_MEAN, IMAGE_STD


print("Performing HTR...")

device = "cuda:0" #or "cpu"

def get_prediction(output):      
    rot_output = torch.squeeze(output).T.detach().cpu().numpy()
    rot_output = np.roll(rot_output, -1, axis=1)
    beams = CTC_DECODER.decode_beams(rot_output)
    beam_prediction = beams[0][0]
    beam_logit_score = beams[0][3] 
    beam_prediction = CTC_DECODER.decode(rot_output)
    return beam_prediction, beam_logit_score
    

class LineImageDataset(Dataset):
    def get_namespace(self, element):
        m = re.match(r'\{.*\}', element.tag)
        return m.group(0)[1:-1] if m else ''    

    def __init__(self, filename, num_to_char, transform=None):
        self.transform = transform       
        self.num_to_char = num_to_char
        self.line_images = []
        self.line_image_filenames = []
        self.image_filenames = []
        self.labels = []
        self.num_labels = []

        dirname = os.path.dirname(filename)
        if dirname == "":
            dirname = "."
        tree = ET.parse(filename)
        ns = {"ns": self.get_namespace(tree.getroot())}
        ET.register_namespace('', ns['ns'])
        root = tree.getroot()

        image_filename = root.find('ns:Page', ns).get('imageFilename')

        for text_region in root.findall('.//ns:TextRegion', ns):
            for lineno, text_line in enumerate(text_region.findall('.//ns:TextLine', ns)):
                if text_line.get("custom") == "type {type:margin;}":
                    continue
                
                line_im_filename = dirname + "/line_{}_{}".format(lineno, os.path.basename(image_filename))
                line_im_filename, _ = os.path.splitext(line_im_filename)
                line_im_filename += ".png"
                self.image_filenames.append(image_filename)    
                self.line_image_filenames.append(line_im_filename)
                try:
                    self.line_images.append(PIL.Image.open(line_im_filename).convert("L"))
                except FileNotFoundError:
                    print("{} not found!".format(line_im_filename))
                    self.line_images.append(PIL.Image.new('L', (256,128))) #Blank image
                                 
                #self.line_images.append(torch.tensor(np.load(line_im_filename.replace(".png", ".npy")), dtype=torch.float32).unsqueeze(0))
                self.labels.append("")
                self.num_labels.append(torch.tensor([]))

                                        
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):                
        image = self.line_images[idx]
        
        if self.transform is not None:
            image = self.transform(image)
       
        return {"image": image, "target": self.num_labels[idx], "text": self.labels[idx]}


val_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(IMAGE_MEAN, IMAGE_STD)
    ])

net = HTRModel()
net.load_state_dict(torch.load("best_HTR.net"))
net.eval()
net.to(device)

input_filename = sys.argv[1]
test_dataset = LineImageDataset(input_filename, NUM_TO_CHAR, transform=val_transform)

predictions = []
for i, line in enumerate(test_dataset):
    output = net(line["image"].unsqueeze(0).to(device))

    #The score is a measure of the model's confidence in its output
    prediction, _ = get_prediction(output)
    predictions.append(prediction)

#Write to XML file
dirname = os.path.dirname(input_filename)
if dirname == "":
    dirname = "."
tree = ET.parse(input_filename)

m = re.match(r'\{.*\}', tree.getroot().tag)
namespace = m.group(0)[1:-1] if m else ''

ns = {"ns": namespace}
ET.register_namespace('', ns['ns'])
root = tree.getroot()

image_filename = root.find('ns:Page', ns).get('imageFilename')
counter = 0
for text_region in root.findall('.//ns:TextRegion', ns):
    for lineno, text_line in enumerate(text_region.findall('.//ns:TextLine', ns)):
        if text_line.get("custom") == "type {type:margin;}":
            #There should be no marginalia text lines in this repository, but we are currently working on adding them
            continue
        
        text_equiv = text_line.find('ns:TextEquiv', ns)
        if text_equiv is None:
            text_equiv = ET.SubElement(text_line, 'TextEquiv')
            
        unicode_elem = text_equiv.find('ns:Unicode', ns)
        if unicode_elem is None:
            unicode_elem = ET.SubElement(text_equiv, 'Unicode')
        
        unicode_elem.text = predictions[counter]
        counter += 1
tree.write(input_filename)

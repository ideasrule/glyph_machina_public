from kraken import serialization
from kraken.lib.vgsl import TorchVGSLModel
from PIL import Image
from kraken import blla
import sys

device = "cuda:0"
model_filename = sys.argv[1]
image_filename = sys.argv[2]
xml_filename = sys.argv[3]

model = TorchVGSLModel.load_model(model_filename)
model.to(device)

def segment_image(image_filename, output_filename):
    im = Image.open(image_filename)

    #If using our custom fork of kraken, you can pass threshold and min_length to blla.segment.  We recommend 0.10 and 100
    res = blla.segment(im, text_direction="horizontal-lr", model=model, device=device) #, threshold=threshold, min_length=min_length)
    xml_contents = serialization.serialize(
        res,
        image_size=im.size,
        template="pagexml",
        template_source='native',
        processing_steps=[
            {'category': 'processing',
             'description': 'Baseline and region segmentation',
             'settings': {'model': model_filename, 'text_direction': 'horizontal-lr'}}],
        sub_line_segmentation=True)
    with open(output_filename, "w") as f:
        f.write(xml_contents)

print("Identifying baselines...")
segment_image(image_filename, xml_filename)

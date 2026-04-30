import pandas as pd
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, GenerationConfig
from torchmetrics.text import CharErrorRate, WordErrorRate
import sys
from dataset import LineDatasetTrOCR

pretrained_model_name = "magistermilitum/tridis_HTR"
finetuned_model_name = sys.argv[1]
df = pd.read_csv("test_list.csv")
device = "cuda:0"

processor = TrOCRProcessor.from_pretrained(pretrained_model_name, use_fast=True)
dataset = LineDatasetTrOCR(root_dir='./',
                                df=df,
                                processor=processor,
                                transform=None)

model = VisionEncoderDecoderModel.from_pretrained(finetuned_model_name)
model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id
model.config.vocab_size = model.config.decoder.vocab_size
model.to(device)

gen_cfg = GenerationConfig.from_pretrained(pretrained_model_name)
gen_cfg.update(eos_token_id = processor.tokenizer.sep_token_id,
               max_length = 64,
               early_stopping = True,
               no_repeat_ngram_size = 3,
               length_penalty = 2.0,
               num_beams = 4)
model.generation_config = gen_cfg

grand_cer_calc = CharErrorRate()
grand_wer_calc = WordErrorRate()
print("XML_filename,CER,WER,num_lines")
for xml_filename in df["xml_filename"].unique():
    cer_calc = CharErrorRate()
    wer_calc = WordErrorRate()
    
    subset = df[df["xml_filename"] == xml_filename]
    for index, row in subset.iterrows():
        generated_ids = model.generate(dataset[index]["pixel_values"].unsqueeze(0).to(device))
        prediction = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
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

#Adapted from https://github.com/NielsRogge/Transformers-Tutorials/blob/master/TrOCR/Fine_tune_TrOCR_on_IAM_Handwriting_Database_using_Seq2SeqTrainer.ipynb

import pandas as pd
import torchvision.transforms as transforms
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, GenerationConfig, Seq2SeqTrainer, Seq2SeqTrainingArguments, default_data_collator
import evaluate
from dataset import LineDatasetTrOCR


train_df = pd.read_csv("train_list.csv")
test_df = pd.read_csv("test_list.csv")

pretrained_model_name = "magistermilitum/tridis_HTR"
train_transform = transforms.Compose(
    [
       transforms.ColorJitter(0.5, 0.5, 0.5, 0.5),
       transforms.RandomAffine(0.7, translate=(0.01, 0.02), scale=(0.98, 1.02)),
       transforms.RandomChoice([
            transforms.RandomAdjustSharpness(2, p=0.5),
            transforms.GaussianBlur(21, (1,6))
        ]),      
    ])

processor = TrOCRProcessor.from_pretrained(pretrained_model_name, use_fast=True)
train_dataset = LineDatasetTrOCR(root_dir='./',
                                 df=train_df,
                                 processor=processor,
                                 transform=train_transform)
eval_dataset = LineDatasetTrOCR(root_dir='./',
                                df=test_df,
                                processor=processor,
                                transform=None)

print("Number of training examples:", len(train_dataset))
print("Number of validation examples:", len(eval_dataset))

model = VisionEncoderDecoderModel.from_pretrained(pretrained_model_name)

# set special tokens used for creating the decoder_input_ids from the labels
model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id
model.config.vocab_size = model.config.decoder.vocab_size

# set beam search parameters
gen_cfg = GenerationConfig.from_pretrained(pretrained_model_name)
gen_cfg.update(eos_token_id = processor.tokenizer.sep_token_id,
               max_length = 64,
               early_stopping = True,
               no_repeat_ngram_size = 3,
               length_penalty = 2.0,
               num_beams = 4)
model.generation_config = gen_cfg

training_args = Seq2SeqTrainingArguments(
    num_train_epochs=30,
    learning_rate=5e-5,
    predict_with_generate=True,
    eval_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    fp16=True, 
    output_dir="./trocr_finetuned",
    #logging_steps=2,
    save_steps=3000,
    save_total_limit=3,
    dataloader_num_workers=8
)

cer_metric = evaluate.load("cer")
wer_metric = evaluate.load("wer")

def compute_metrics(pred):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions

    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = processor.tokenizer.pad_token_id
    label_str = processor.batch_decode(labels_ids, skip_special_tokens=True)

    cer = cer_metric.compute(predictions=pred_str, references=label_str)
    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    return {"char_acc": 1-cer, "word_acc": 1-wer}

trainer = Seq2SeqTrainer(
    model=model,
    processing_class=processor.image_processor,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=default_data_collator,
  )
trainer.train()

print(trainer.evaluate())

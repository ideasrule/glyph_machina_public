This repository contains the code necessary to reproduce the results in our paper, "Democratizing the medieval English legal tradition".  All training data are from the [AALT](https://aalt.law.uh.edu/).  Note that the version of the pipeline on [Glyph Machina](https://glyphmachina.com/) achieves higher segmentation and transcription accuracy than the version in this repository.  This is because the website incorporates the many improvements we have made after submitting the paper, including:

1. The segmentation model is trained on 2000 images, instead of the 173 in this repository
2. The HTR model is first trained on the Latin transcriptions in the TRIDIS dataset, then fine-tuned on our dataset. This decreases the Character Error Rate by 0.8% (to 5.2%) and the Word Error Rate by 1.6% (to 16.2%).
3. Actually, the HTR model is fine-tuned on a dataset 60% larger, consisting of transcriptions from a wider range of sources.  This further decreases the CER to 4.9% and the WER to 15.5%.


# Dependencies
If you want to retrain the models that make up the pipeline, you will need to install kraken 6 (not 7), pandas, pytorch, lightning, pyctcdecode, KenLM, torchmetrics, and PIL.  If you are interested in training or using the transformer-based TrOCR, you will also need to install transformers, datasets, evaluate, and accelerate.

Most of these dependencies can be installed with:

    conda env create -f environment.yml

We have also created a [Docker image](https://hub.docker.com/repository/docker/ideasrule/glyph_machina) with the full development environment needed to run all the code.  The image also contains emacs, X libraries, and the PageXML viewer and editor [visual-page-editor](https://github.com/buzzcauldron/visual-page-editor).

# Training
To train the image segmentation neural network:

1. Install [kraken](https://github.com/mittagessen/kraken)
2. Run segtrain.sh.  kraken will train on the PageXML files in train_data/ and test_data/.  Instead of training from scratch, it will start from pretrained_backbone_blla.mlmodel--the same blla.mlmodel model that ships with kraken, except that input images are resized to have a width of 1800 pixels instead of a height of 1800 pixels.
3. Choose the model which maximizes val_mean_iu while not being too horrible in val_freq_iu.  Rename the best model seg.mlmodel.


To train the handwriting recognition neural network:

1. Run run_line_image_generator.py on all PageXML files in train_data/ and test_data/:

```
	python run_line_image_generator.py train_data/*.xml > train_list.csv
	python run_line_image_generator.py test_data/*.xml > test_list.csv
```

2. Run train_on_line_list.py.  This will read train_list.csv and test_list.csv, and write best_HTR.net.
3. Dump all the training text into a file and train a bigram model using [KenLM](https://github.com/kpu/kenlm), which you must install beforehand:
```
	awk -F',' '{print $4}' train_list.csv | tail -n +2 > all_training_text.txt
	KENLM_DIR/bin/lmplz -o 2 < all_training_text.txt > bigram_model.arpa
```
4. Run "python eval_on_line_list.py best_HTR.net test_list.csv" to compute the CER and WER on every case in the test set.  This will use bigram_model.arpa.

To train the transformer-based (TrOCR) version of the handwriting recognition neural network (not recommended):

1. python finetune_TrOCR.py
2. python eval_TrOCR.py trocr_finetuned/checkpoint-13530/

Of course, modify the model directory in the second step if you change the number of training steps.


# Running end-to-end

To run handwriting recognition on an image:

1. python run_segmenter.py image.JPG image.xml.  This uses seg.mlmodel.  We have found that setting threshold=0.10 and min_length=100 in kraken works better than the default of threshold=0.17 and min_length=5; however, kraken does not expose these parameters to the user.  You can leave the default values, hack kraken (KRAKEN_PATH/kraken/lib/segmentation.py:vectorize_lines) to change the default values, or use our [custom fork of kraken](https://github.com/ideasrule/kraken), which does expose these two arguments to the user.  Our Docker image contains a hacked version of kraken.

2. python run_line_image_generator.py image.xml > /dev/null
3. python run_htr.py image.xml.  This uses best_HTR.net.
4. (Optional) Set your GEMINI_API_KEY environment variable, then "python run_gemini.py image.xml corrected_image.xml"

image.xml and corrected_image.xml will be PageXML files with the transcriptions embedded.

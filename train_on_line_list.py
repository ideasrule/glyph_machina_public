from torch import optim, nn, utils
import pytorch_lightning as L
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import loggers as pl_loggers
import torch
torch.manual_seed(42)
print("Using GPU", torch.cuda.is_available())
import numpy as np
from torchmetrics.text import CharErrorRate, WordErrorRate
import torchvision.transforms.v2 as transforms

from dataset import LineImageDataset
from model import HTRModel, NUM_TO_CHAR, CHAR_TO_NUM, IMAGE_MEAN, IMAGE_STD

train_transform = transforms.Compose(
    [
        transforms.ColorJitter(0.5, 0.5, 0.5, 0.5),
        transforms.RandomAffine(0.7, translate=(0.01, 0.02), scale=(0.98, 1.02)),
        transforms.RandomChoice([
            transforms.RandomAdjustSharpness(2, p=0.5),
            transforms.GaussianBlur(21, (1,3))
        ]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[IMAGE_MEAN], std=[IMAGE_STD]),
    ])

val_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=[IMAGE_MEAN], std=[IMAGE_STD]),
    ])

#We don't have a dedicated validation set, so we use "val" and "test" interchangeably
train_dataset = LineImageDataset("train_list.csv", CHAR_TO_NUM, transform=train_transform)
val_dataset = LineImageDataset("test_list.csv", CHAR_TO_NUM, transform=val_transform)

net = HTRModel()
net.eval()

class LatinTranscriber(L.LightningModule):
    def __init__(self, net, num_to_char):
        super().__init__()
        self.num_to_char = num_to_char
        self.cer_calc = CharErrorRate()
        self.wer_calc = WordErrorRate()
        self.train_cer_calc = CharErrorRate()
        self.train_wer_calc = WordErrorRate()
        self.net = net

    def get_loss(self, batch, batch_idx):
        target = batch["target"]
        target_length = batch["target"].shape[1]
        input = batch["image"]
        output = self.net(input)
        output_length = output.shape[-1]

        loss_func = nn.CTCLoss(reduction='sum', zero_infinity=True)
        loss = loss_func(output.permute(2,0,1), target, (output_length,), (target_length,))
        return loss, output

    def on_train_epoch_start(self):
        self.train_cer_calc.reset()
        self.train_wer_calc.reset()


    def _get_current_lr(self):
        for param_group in self.trainer.optimizers[0].param_groups:
            return param_group['lr']

    def on_train_epoch_end(self):
        char_accuracy = 1 - self.train_cer_calc.compute()
        word_accuracy = 1 - self.train_wer_calc.compute()
        lr = self._get_current_lr()

        self.log("train_char_acc", char_accuracy)
        self.log("train_word_acc", word_accuracy)
        self.log('lr-Adam', lr)

    def training_step(self, batch, batch_idx):
        assert self.net.training
        loss, output = self.get_loss(batch, batch_idx)
        prediction, truth = self.get_prediction_and_truth(output, batch["target"])
        self.train_cer_calc.update(truth, prediction)
        self.train_wer_calc.update(truth, prediction)
        self.log("train_loss", loss)
        return loss

    def get_prediction(self, output):
        #Greedy decoding
        rot_output = torch.squeeze(output).T.detach().cpu().numpy()
        rot_output = np.roll(rot_output, -1, axis=1)

        labels = torch.argmax(torch.squeeze(output), axis=0).cpu().numpy()
        prediction = ""
        for i in range(len(labels)):
            label = labels[i]
            if label != 0 and (i==0 or label != labels[i-1]):
                prediction += self.num_to_char[label]
        return prediction

    def get_prediction_and_truth(self, output, target):
        prediction = self.get_prediction(output)
        target = np.atleast_1d(torch.squeeze(target).cpu().numpy())
        truth = ''.join([self.num_to_char[target[i].item()] for i in range(len(target))])
        return prediction, truth


    def validation_step(self, batch, batch_idx):
        assert not self.net.training
        assert batch["target"].shape[0] == 1
        loss, output = self.get_loss(batch, batch_idx)
        prediction, truth = self.get_prediction_and_truth(output, batch["target"])

        self.cer_calc.update(truth, prediction)
        self.wer_calc.update(truth, prediction)

        # Get tensorboard logger
        if batch_idx < 16:
            tb_logger = None
            for logger in self.trainer.loggers:
                if isinstance(logger, pl_loggers.TensorBoardLogger):
                    tb_logger = logger.experiment
                    break

            tb_logger.add_image(f'Validation #{batch_idx}, target: {truth}', batch['image'][0], self.global_step, dataformats="CHW")
            tb_logger.add_text(f'Validation #{batch_idx}, target: {truth}', prediction, self.global_step)

        return loss

    def on_validation_epoch_start(self):
        self.cer_calc.reset()
        self.wer_calc.reset()

    def on_validation_epoch_end(self):
        char_accuracy = 1 - self.cer_calc.compute()
        word_accuracy = 1 - self.wer_calc.compute()
        print("Epoch, char acc, word acc:", self.current_epoch, round(char_accuracy.item(), 4), round(word_accuracy.item(), 4))
        self.log("val_char_acc", char_accuracy)
        self.log("val_word_acc", word_accuracy)


    def configure_optimizers(self):        
        optimizer = optim.AdamW(self.parameters(), lr=1e-3, weight_decay=1e-2)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "name": "lr-scheduler",
                "scheduler": optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.34, min_lr=1e-5, patience=10),
                "monitor": "val_word_acc",
                "frequency": 1
            },
        }

transcriber = LatinTranscriber(net, NUM_TO_CHAR)

train_loader = utils.data.DataLoader(train_dataset, num_workers=8)
valid_loader = utils.data.DataLoader(val_dataset, num_workers=4)

net.train()
checkpoint_callback = ModelCheckpoint(
    monitor="val_word_acc", mode="max", dirpath="./", filename="ocr"
)

trainer = L.Trainer(accumulate_grad_batches=1, max_epochs=250, enable_progress_bar=True, callbacks=[checkpoint_callback])
trainer.fit(transcriber, train_loader, valid_loader)
trainer.validate(ckpt_path="best", dataloaders=valid_loader)

best_transcriber = LatinTranscriber.load_from_checkpoint(
    checkpoint_callback.best_model_path,
    net=net,
    num_to_char=NUM_TO_CHAR)
torch.save(best_transcriber.net.state_dict(), "best_HTR.net")

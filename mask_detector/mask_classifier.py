from argparse import ArgumentParser

import torch
from torch.nn.functional import binary_cross_entropy
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.metrics import Accuracy
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.metrics import accuracy_score

from model import BasicCNN
from dataset import MaskDataset
from utils import train_val_test_split


class MaskClassifier(LightningModule):
    def __init__(self, net, learning_rate=1e-3):
        super().__init__()
        self.net = net
        self.learning_rate = learning_rate
        self.accuracy = Accuracy()

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self.net(x)
        loss = binary_cross_entropy(out, y)

        return {'loss': loss, 'accuracy': self.accuracy(out, y)}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        out = self.net(x)
        loss = binary_cross_entropy(out, y)

        return {'loss': loss, 'accuracy': self.accuracy(out, y)}

    def test_step(self, batch, batch_idx):
        x, y = batch
        out = self.net(x)
        loss = binary_cross_entropy(out, y)

        return {'loss': loss, 'accuracy': self.accuracy(out, y)}

    def configure_optimizers(self):
        # self.hparams available because we called self.save_hyperparameters()
        return Adam(self.parameters(), lr=self.learning_rate)


def cli_main():
    seed_everything(1234)

    # ------------
    # data
    # ------------
    dataset = MaskDataset(csv_file='data/dataframe/mask_df.csv')
    ds_train, ds_validate, ds_test = train_val_test_split(
        dataset, train_ratio=0.8, validate_ratio=0.1, test_ratio=0.1)

    train_loader = DataLoader(ds_train, batch_size=32)
    val_loader = DataLoader(ds_validate, batch_size=32)
    test_loader = DataLoader(ds_test, batch_size=32)

    # ------------
    # model
    # ------------
    net = BasicCNN()
    model = MaskClassifier(net, learning_rate=0.0001)

    # ------------
    # training
    # ------------
    gpus = 1 if torch.cuda.is_available() else 0

    trainer = Trainer(gpus=gpus, max_epochs=3)
    trainer.fit(model, train_loader, val_loader)

    # ------------
    # testing
    # ------------
    result = trainer.test(test_dataloaders=test_loader)
    print(result)


if __name__ == '__main__':
    cli_main()

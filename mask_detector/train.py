from argparse import ArgumentParser

import torch
from torch.nn.functional import binary_cross_entropy
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.metrics import accuracy_score

from model import BasicCNN
from dataset import MaskDataset
from utils import train_val_test_split


class MaskClassifier(LightningModule):
    def __init__(self, net, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.net = net

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self.net(x)

        loss = binary_cross_entropy(out, y)

        self.log('train_loss', loss, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        out = self.net(x)
        loss = binary_cross_entropy(out, y)

        self.log('valid_loss', loss, on_step=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        out = self.net(x)
        loss = binary_cross_entropy(out, y)

        _, out = torch.max(out, dim=1)
        val_acc = accuracy_score(out.cpu(), y.cpu())
        val_acc = torch.tensor(val_acc)

        return {'test_loss': loss, 'test_acc': val_acc}

    def configure_optimizers(self):
        # self.hparams available because we called self.save_hyperparameters()
        return Adam(self.parameters(), lr=self.hparams.learning_rate)


def cli_main():
    seed_everything(1234)

    # ------------
    # data
    # ------------
    dataset = MaskDataset(csv_file='data/dataframe/mask_df.csv')
    ds_train, ds_validate, ds_test = train_val_test_split(
        dataset, train_ratio=0.8, validate_ratio=0.1, test_ratio=0.1)

    train_loader = DataLoader(ds_train, batch_size=128)
    val_loader = DataLoader(ds_validate, batch_size=128)
    test_loader = DataLoader(ds_test, batch_size=128)

    # ------------
    # model
    # ------------
    net = BasicCNN()
    model = MaskClassifier(net, learning_rate=0.0001)

    # ------------
    # training
    # ------------
    checkpoint_callback = ModelCheckpoint(
        verbose=True,
        monitor='test_acc',
        mode='max'
    )
    trainer = Trainer(max_epochs=1, checkpoint_callback=checkpoint_callback)
    trainer.fit(model, train_loader, val_loader)

    # ------------
    # testing
    # ------------
    result = trainer.test(test_dataloaders=test_loader)
    print(result)


if __name__ == '__main__':
    cli_main()

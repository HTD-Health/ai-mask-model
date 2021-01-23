from argparse import ArgumentParser

import torch
from torch.nn.functional import binary_cross_entropy
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.metrics import Accuracy
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.metrics import accuracy_score

from utils import train_val_test_split
from models.basic_cnn import BasicCNN
from datasets.masked_face_net import MaskedFaceNetDataset


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

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--learning_rate', type=float, default=1e-3)
        return parser


def cli_main():
    seed_everything(1234)

    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument('--batch_size', default=32, type=int)
    parser = Trainer.add_argparse_args(parser)
    parser = MaskClassifier.add_model_specific_args(parser)
    args = parser.parse_args()

    # ------------
    # data
    # ------------
    dataset = MaskedFaceNetDataset(csv_file='data/dataframe/mask_df.csv')
    ds_train, ds_validate, ds_test = train_val_test_split(
        dataset, train_ratio=0.8, validate_ratio=0.1, test_ratio=0.1)

    train_loader = DataLoader(ds_train, batch_size=args.batch_size)
    val_loader = DataLoader(ds_validate, batch_size=args.batch_size)
    test_loader = DataLoader(ds_test, batch_size=args.batch_size)

    # ------------
    # model
    # ------------
    net = BasicCNN()
    model = MaskClassifier(net, learning_rate=args.learning_rate)

    # ------------
    # training
    # ------------
    trainer = Trainer.from_argparse_args(args)
    trainer.fit(model, train_loader, val_loader)

    # ------------
    # testing
    # ------------
    result = trainer.test(test_dataloaders=test_loader)
    print(result)


if __name__ == '__main__':
    cli_main()

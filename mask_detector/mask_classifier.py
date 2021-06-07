from argparse import ArgumentParser
from datetime import datetime

from pytorch_lightning import (LightningModule, Trainer, seed_everything)
from pytorch_lightning.metrics import Recall
from torch.nn.functional import binary_cross_entropy
from torch.optim import Adam
from torch.utils.data import DataLoader

from mask_detector.datasets.masked_face_net import MaskedFaceNetDataset
from mask_detector.models.mobile_net_v2 import MobileNetV2
from mask_detector.utils import train_val_test_split


class MaskClassifier(LightningModule):
    # TODO: add docstrings
    def __init__(self, net, learning_rate=0.001):
        super().__init__()
        self.net = net
        self.learning_rate = learning_rate
        self.recall = Recall()

    def forward(self, x):
        # TODO: add docstrings
        return self.net(x)

    # TODO: resolve lack of usage of batch_idx
    def training_step(self, batch, batch_idx):
        # TODO: add docstrings
        x, y = batch
        out = self.net(x)
        loss = binary_cross_entropy(out, y)
        recall = self.recall(out, y)

        self.log('train_loss', loss, on_step=False, on_epoch=True)
        self.log('train_recall', recall, on_step=False, on_epoch=True)

        return loss

    # TODO: resolve lack of usage of batch_idx
    def validation_step(self, batch, batch_idx):
        # TODO: add docstrings
        x, y = batch
        out = self.net(x)
        loss = binary_cross_entropy(out, y)
        recall = self.recall(out, y)

        self.log('val_loss', loss, on_step=False, on_epoch=True)
        self.log('val_recall', recall, on_step=False, on_epoch=True)

        return loss

    # TODO: resolve lack of usage of batch_idx
    def test_step(self, batch, batch_idx):
        # TODO: add docstrings
        x, y = batch
        out = self.net(x)
        loss = binary_cross_entropy(out, y)
        recall = self.recall(out, y)

        self.log('test_loss', loss, on_step=False, on_epoch=True)
        self.log('test_recall', recall, on_step=False, on_epoch=True)

        return loss

    def configure_optimizers(self):
        # TODO: add docstrings
        return Adam(self.parameters(), lr=self.learning_rate)

    @staticmethod
    def add_model_specific_args(parent_parser):
        # TODO: add docstrings
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--learning_rate', type=float, default=0.001)
        return parser


def cli_main():
    # TODO: add docstrings
    seed_everything(1234)

    # ------------
    # args
    # ------------
    parser = ArgumentParser()

    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--image_size', default=120, type=int)

    parser = Trainer.add_argparse_args(parser)
    parser = MaskClassifier.add_model_specific_args(parser)
    args = parser.parse_args()

    # ------------
    # data
    # ------------
    dataset = MaskedFaceNetDataset(
        csv_file='./data/dataframe/mask_df.csv', image_size=args.image_size)
    ds_train, ds_validate, ds_test = train_val_test_split(
        dataset, train_ratio=0.8, validate_ratio=0.1, test_ratio=0.1)

    train_loader = DataLoader(ds_train, batch_size=args.batch_size)
    val_loader = DataLoader(ds_validate, batch_size=args.batch_size)
    test_loader = DataLoader(ds_test, batch_size=args.batch_size)

    # ------------
    # model
    # ------------
    net = MobileNetV2()
    model = MaskClassifier(net, learning_rate=args.learning_rate)

    # ------------
    # training
    # ------------
    trainer = Trainer.from_argparse_args(args)
    result = trainer.fit(model, train_loader, val_loader)

    # ------------
    # testing
    # ------------
    result = trainer.test(test_dataloaders=test_loader)
    print(result)

    # ------------
    # saving model checkpoint
    # ------------
    now = datetime.now()
    trainer.save_checkpoint("../model_checkpoints/model_checkpoint_" + now.strftime("%d/%m/%Y_%H:%M:%S"))


if __name__ == '__main__':
    cli_main()

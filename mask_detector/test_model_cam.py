from argparse import ArgumentParser

import torch
from torch.nn.functional import binary_cross_entropy
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.metrics import Recall
from pytorch_lightning.callbacks import ModelCheckpoint

import pickle
import gradio as gr

from utils import train_val_test_split
from models.basic_cnn import BasicCNN
from models.mobile_net_v2 import MobileNetV2
from datasets.masked_face_net import MaskedFaceNetDataset


class MaskClassifier(LightningModule):
    def __init__(self, net, learning_rate=0.001):
        super().__init__()
        self.net = net
        self.learning_rate = learning_rate
        self.recall = Recall()

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self.net(x)
        loss = binary_cross_entropy(out, y)
        recall = self.recall(out, y)

        self.log('train_loss', loss, on_step=False, on_epoch=True)
        self.log('train_recall', recall, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        out = self.net(x)
        loss = binary_cross_entropy(out, y)
        recall = self.recall(out, y)

        self.log('val_loss', loss, on_step=False, on_epoch=True)
        self.log('val_recall', recall, on_step=False, on_epoch=True)

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        out = self.net(x)
        loss = binary_cross_entropy(out, y)
        recall = self.recall(out, y)

        self.log('test_loss', loss, on_step=False, on_epoch=True)
        self.log('test_recall', recall, on_step=False, on_epoch=True)

        return loss

    def configure_optimizers(self):
        # self.hparams available because we called self.save_hyperparameters()
        return Adam(self.parameters(), lr=self.learning_rate)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--learning_rate', type=float, default=0.001)
        return parser


def test():
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
    # dataset = MaskedFaceNetDataset(
    #     csv_file='data/dataframe/mask_df.csv', image_size=args.image_size)
    # ds_train, ds_validate, ds_test = train_val_test_split(
    #     dataset, train_ratio=0.8, validate_ratio=0.1, test_ratio=0.1)

    # with open('test_data.pickle', 'wb') as handle:
    #     pickle.dump(ds_test, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # return

    with open('test_data.pickle', 'rb') as handle:
        ds_test = pickle.load(handle)

    # train_loader = DataLoader(ds_train, batch_size=args.batch_size)
    # val_loader = DataLoader(ds_validate, batch_size=args.batch_size)
    test_loader = DataLoader(ds_test, batch_size=args.batch_size)

    # ------------
    # model
    # ------------
    net = MobileNetV2()
    model = MaskClassifier.load_from_checkpoint('lightning_logs/version_0/checkpoints/epoch=7-step=759.ckpt', net=net)

    # print(model)

    def classify_mask(img):
        model.eval()
        return img
        # with torch.no_grad():
        #     for y_pred, y_exp in zip(ys_pred, ys_exp):
        #         print(f'Expected: {y_exp.item():.0f}   Predicted: {y_pred.item():.1f}')

    webcam = gr.inputs.Image(shape=(200, 200), source="webcam")
    gr.Interface(fn=classify_mask, inputs=webcam, outputs="image").launch()


if __name__ == '__main__':
    # cli_main()
    test()

import os
import logging

import hydra
from omegaconf import OmegaConf, DictConfig

from pytorch_lightning import Trainer, loggers
import torch

from src import Conv1dAutoEncoder, LSTMAutoEncoder, TransactionDataModuleNewData, LSTMAutoEncoderEmbed
from src.utils.logging_utils import get_logger
from src.preprocessing.new_data_preprop import preprocessing

logger = get_logger(name=__name__)


def test_lstm_network(train_dataset):
    model = LSTMAutoEncoder(40, 3)
    logger = loggers.TensorBoardLogger('lightning_logs_new', 'lstm')
    trainer = Trainer(gpus=0, max_epochs=20, logger=logger)
    #dm = TransactionDataModule(train_dataset, test_dataset, drop_time=
    dm = TransactionDataModuleNewData(train_dataset)
    trainer.fit(model, dm)
    trainer.test(model, dm)


def test_lstm_network_embed(train_dataset):
    model = LSTMAutoEncoderEmbed(17, 4)
    logger = loggers.TensorBoardLogger('lightning_logs_new', 'lstm')
    trainer = Trainer(gpus=0, max_epochs=20, logger=logger)
    #dm = TransactionDataModule(train_dataset, test_dataset, drop_time=
    dm = TransactionDataModuleNewData(train_dataset)
    trainer.fit(model, dm)
    trainer.test(model, dm)


def test_cae_network(train_dataset):
    model = Conv1dAutoEncoder(1, 8)
    logger = loggers.TensorBoardLogger('lightning_logs', 'cae')
    trainer = Trainer(gpus=1, max_epochs=4, logger=logger)
    dm = TransactionDataModuleNewData(train_dataset)

    trainer.fit(model, dm)
    # trainer.test(model, dm)


def test_lstm_freeze(train_dataset):
    model = LSTMAutoEncoderEmbed(17, 4)
    checkpoint = torch.load('.\\lightning_logs_new\\lstm\\version_4\\checkpoints\\epoch=19-step=87780.ckpt')
    model.load_state_dict(checkpoint['state_dict'])
    logger = loggers.TensorBoardLogger('lightning_logs_new', 'lstm')
    trainer = Trainer(gpus=0, max_epochs=20, logger=logger)
    #dm = TransactionDataModule(train_dataset, test_dataset, drop_time=
    dm = TransactionDataModuleNewData(train_dataset)
    #trainer.fit(model, dm)
    trainer.test(model, dm)


@hydra.main(config_path='config', config_name='config', version_base=None)
def main(cfg: DictConfig) -> None:
    if cfg['dataset']['name'] == 'new_data':
        preprocessing(cfg['dataset'])



if __name__ == '__main__':
    # data_folder = '.\\data\\normal\\'
    # test_cae_network(data_folder)
    # test_lstm_freeze(data_folder)
    main()

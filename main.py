import hydra
from omegaconf import DictConfig

from src.utils.logging_utils import get_logger
from src import train_tr2vec, train_lstm

logger = get_logger(name=__name__)


@hydra.main(config_path='config', config_name='config', version_base=None)
def main(cfg: DictConfig) -> None:
    # TODO Fix this
    with open('api_token.txt') as f:
        api_token = f.read()

    mode: str = cfg['task'].lower()
    logger.info(f'Working mode - {mode}')
    if mode == 'tr2vec':
        train_tr2vec(cfg['dataset'], cfg['embed_model'], api_token)
    elif mode == 'lstm':
        train_lstm(cfg['dataset'], cfg['autoencoder'], api_token)


if __name__ == '__main__':
    main()

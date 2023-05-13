import hydra
from omegaconf import DictConfig

from src.utils.logging_utils import get_logger
from src.learning import train_embed_model, train_autoencoder

logger = get_logger(name=__name__)


@hydra.main(config_path='config', config_name='config', version_base=None)
def main(cfg: DictConfig) -> None:
    # TODO Fix this
    with open('api_token.txt') as f:
        api_token = f.read()

    mode: str = cfg['task'].lower()
    logger.info(f'Working mode - {mode}')
    if mode == 'embed_model':
        train_embed_model(cfg['embed_model']['name'])(
            cfg['dataset'], cfg['embed_model'], api_token
        )
    elif mode == 'autoencoder':
        train_autoencoder(cfg['autoencoder']['name'])(
            cfg['dataset'], cfg['autoencoder'], api_token
        )


if __name__ == '__main__':
    main()

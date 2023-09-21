import hydra
from omegaconf import DictConfig, OmegaConf

from src.utils.logging_utils import get_logger
from src.learning import train_autoencoder

logger = get_logger(name=__name__)


@hydra.main(config_path="config", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    train_autoencoder(cfg)


if __name__ == "__main__":
    main()

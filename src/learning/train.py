from .embed_models import train_tr2vec, train_coles
from .autoencoders import train_lstm

def train_embed_model(name: str):
    if name == 'tr2vec':
        return train_tr2vec
    elif name == 'coles':
        return train_coles
    
def train_autoencoder(name: str):
    if name == 'lstm':
        return train_lstm

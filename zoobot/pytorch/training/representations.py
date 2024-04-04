import logging
import pytorch_lightning as pl

from timm import create_model


class ZoobotEncoder(pl.LightningModule):

    def __init__(self, encoder):
        super().__init__()
        logging.info('ZoobotEncoder: using provided in-memory encoder')
        self.encoder = encoder  # plain pytorch module e.g. Sequential


    def forward(self, x):
        if isinstance(x, list) and len(x) == 1:
            return self(x[0])
        return self.encoder(x)
    
    @classmethod
    def load_from_name(cls, name: str):
        """
        e.g. ZoobotEncoder.load_from_name('hf_hub:mwalmsley/zoobot-encoder-convnext_nano')
        Args:
            name (str): huggingface hub name to load

        Returns:
            nn.Module: timm model
        """
        timm_model = create_model(name, pretrained=True)
        return cls(timm_model)
    
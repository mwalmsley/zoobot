import pytorch_lightning as pl

class ZoobotEncoder(pl.LightningModule):
    # very simple wrapper to turn pytorch model into lightning module
    # useful when we want to use lightning to make predictions with our encoder
    # (i.e. to get representations)

    def __init__(self, encoder, pyramid=False) -> None:
        super().__init__()
        self.encoder = encoder  # plain pytorch module e.g. Sequential
        if pyramid:
            raise NotImplementedError('Will eventually support resetting timm classifier to get FPN features')

    def forward(self, x):
        return self.encoder(x)

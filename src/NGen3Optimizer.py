import torch
from transformers.optimization import AdamW


class NGen3Optimizer:
    """
    NGen-3 Optimizer.

    This class implements the AdamW optimizer for NGen-3.
    """

    def __init__(self, model, learning_rate, weight_decay):
        """
        Initializes the AdamW optimizer.

        Args:
            model (nn.Module): NGen-3 model.
            learning_rate (float): Learning rate.
            weight_decay (float): Weight decay.
        """
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

    def get_optimizer(self):
        """
        Returns the AdamW optimizer.

        Returns:
            AdamW: AdamW optimizer.
        """
        return AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

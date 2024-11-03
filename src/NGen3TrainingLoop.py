import torch
from torch.utils.data import DataLoader
from transformers.optimization import AdamW, get_linear_schedule_with_warmup

from NGen3Model import build_ngen3_model, config

class NGen3TrainingLoop:
    """
    NGen-3 Training Loop.

    This class implements the training loop for NGen-3.
    """

    def __init__(self, model, device, train_dataset, batch_size, epochs, learning_rate, weight_decay):
        """
        Initializes the training loop.

        Args:
            model (nn.Module): NGen-3 model.
            device (torch.device): Device.
            train_dataset (Dataset): Training dataset.
            batch_size (int): Batch size.
            epochs (int): Number of epochs.
            learning_rate (float): Learning rate.
            weight_decay (float): Weight decay.
        """
        self.model = model
        self.device = device
        self.train_dataset = train_dataset
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

    def train(self):
        """
        Trains the model.
        """
        train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        optimizer = AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(train_dataloader) * self.epochs)

        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0
            for batch in train_dataloader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                optimizer.zero_grad()

                outputs = self.model(input_ids, attention_mask)
                loss = torch.nn.CrossEntropyLoss()(outputs, labels)

                loss.backward()
                optimizer.step()
                scheduler.step()

                total_loss += loss.item()

            print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_dataloader)}")


# Example usage
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_dataset = ...  # Load your dataset
model = build_ngen3_model(config)
training_loop = NGen3TrainingLoop(model, device, train_dataset, batch_size=32, epochs=5, learning_rate=1e-5, weight_decay=0.01)
training_loop.train()
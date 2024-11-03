import torch
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from rouge import Rouge


from NGen3Model import build_ngen3_model, config
from NGen3Tokenizer import tokenizer
from NGen3DataLoading import load_data

class NGen3EvaluationMetrics:
    """
    NGen-3 Evaluation Metrics.

    This class implements the evaluation metrics for NGen-3.
    """

    def __init__(self, model, device, data_loader):
        """
        Initializes the evaluation metrics.

        Args:
            model (nn.Module): NGen-3 model.
            device (torch.device): Device.
            data_loader (DataLoader): Data loader.
        """
        self.model = model
        self.device = device
        self.data_loader = data_loader

    def evaluate(self):
        """
        Evaluates the model.

        Returns:
            dict: Evaluation metrics.
        """
        self.model.eval()
        total_loss = 0
        total_correct = 0
        total_labels = []
        total_preds = []

        with torch.no_grad():
            for batch in self.data_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.model(input_ids, attention_mask)
                loss = torch.nn.CrossEntropyLoss()(outputs, labels)

                total_loss += loss.item()
                _, predicted = torch.max(outputs.scores, dim=1)
                total_correct += (predicted == labels).sum().item()
                total_labels.extend(labels.cpu().numpy())
                total_preds.extend(predicted.cpu().numpy())

        accuracy = accuracy_score(total_labels, total_preds)
        f1 = f1_score(total_labels, total_preds, average="macro")
        rouge = Rouge()
        rouge_score = rouge.get_scores(total_preds, total_labels, avg=True)

        return {
            "loss": total_loss / len(self.data_loader),
            "accuracy": accuracy,
            "f1": f1,
            "rouge": rouge_score,
            "classification_report": classification_report(total_labels, total_preds),
            "confusion_matrix": confusion_matrix(total_labels, total_preds),
        }


# Example usage
model = build_ngen3_model(config)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_loader = load_data("test.txt", tokenizer, batch_size=32, max_len=512)
evaluation_metrics = NGen3EvaluationMetrics(model, device, data_loader)
metrics = evaluation_metrics.evaluate()
print(metrics)
from datasets import load_dataset, Dataset
from torch import nn
from torch.utils.data import DataLoader



class Evaluate:
    def __init__(self, batch_size: int, data_collator: callable, metrics_fn : callable, pred_fn: callable):
        self.data_collator = data_collator
        self.metrics_fn = metrics_fn
        self.batch_size = batch_size
        self.pred_fn = pred_fn

    def __call__(self, model : nn.Module, data):
        loader = DataLoader(
            data,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.data_collator
        )
        all_preds = []
        all_targets = []

        model.eval()
        for batch in loader:
            labels = batch.pop("labels")
            logits = self.pred_fn(model,batch)
            all_preds.append(logits)
            all_targets.append(labels)
        preds = torch.cat(all_preds)
        targets = torch.cat(all_targets)

        return self.metrics_fn(preds, targets)
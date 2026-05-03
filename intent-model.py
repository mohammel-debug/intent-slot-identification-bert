import torch
import torch.nn as nn
from transformers import BertModel

class text_classification_model(nn.Module):

    def __init__(self, num_intents, dropout=0.1):
        super().__init__()
        self.bert       = BertModel.from_pretrained('bert-base-uncased')
        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Linear(768, num_intents)

    def fit(self, train_loader, dev_loader, epochs=5, lr=3e-5):
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr)
        loss_fn   = nn.CrossEntropyLoss()

        self.to(device)

        for epoch in range(1, epochs+1):
            # ── Training ──
            self.train()
            total_loss = 0
            for batch in train_loader:
                input_ids      = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                token_type_ids = batch['token_type_ids'].to(device)
                intent_label   = batch['intent_label'].to(device)

                optimizer.zero_grad()
                output = self.bert(input_ids, attention_mask, token_type_ids)
                cls_vector = self.dropout(output.pooler_output)  # (B, 768)
                logits     = self.classifier(cls_vector)          # (B, 26)
                loss       = loss_fn(logits, intent_label)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            # ── Validation ──
            acc = self.predict(dev_loader, evaluate=True)
            print(f"Epoch {epoch} | Loss: {total_loss/len(train_loader):.4f} | Dev Acc: {acc:.4f}")

    def predict(self, loader, evaluate=False):
        self.eval()
        all_preds, all_true = [], []
        with torch.no_grad():
            for batch in loader:
                input_ids      = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                token_type_ids = batch['token_type_ids'].to(device)
                output     = self.bert(input_ids, attention_mask, token_type_ids)
                cls_vector = self.dropout(output.pooler_output)
                logits     = self.classifier(cls_vector)
                preds      = logits.argmax(dim=-1).cpu().tolist()
                all_preds.extend(preds)
                if evaluate:
                    all_true.extend(batch['intent_label'].tolist())

        if evaluate:
            from sklearn.metrics import accuracy_score
            return accuracy_score(all_true, all_preds)
        return all_preds


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device:', device)

intent_model = text_classification_model(num_intents=NUM_INTENTS)

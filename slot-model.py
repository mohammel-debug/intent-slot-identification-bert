!pip install seqeval -q

class sequence_labeling_model(nn.Module):

    def __init__(self, num_slots, dropout=0.1):
        super().__init__()
        self.bert       = BertModel.from_pretrained('bert-base-uncased')
        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Linear(768, num_slots)

    def fit(self, train_loader, dev_loader, epochs=5, lr=3e-5):
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr)
        loss_fn   = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

        self.to(device)

        for epoch in range(1, epochs+1):
            # ── Training ──
            self.train()
            total_loss = 0
            for batch in train_loader:
                input_ids      = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                token_type_ids = batch['token_type_ids'].to(device)
                slot_labels    = batch['slot_labels'].to(device)

                optimizer.zero_grad()
                output       = self.bert(input_ids, attention_mask, token_type_ids)
                token_vecs   = self.dropout(output.last_hidden_state)  # (B, 64, 768)
                logits       = self.classifier(token_vecs)              # (B, 64, 128)

                # Flatten for loss: (B*64, 128) vs (B*64,)
                loss = loss_fn(
                    logits.view(-1, logits.shape[-1]),
                    slot_labels.view(-1)
                )
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            # ── Validation ──
            f1 = self.predict(dev_loader, evaluate=True)
            print(f"Epoch {epoch} | Loss: {total_loss/len(train_loader):.4f} | Dev Slot F1: {f1:.4f}")

    def predict(self, loader, evaluate=False):
        self.eval()
        all_preds, all_true = [], []
        with torch.no_grad():
            for batch in loader:
                input_ids      = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                token_type_ids = batch['token_type_ids'].to(device)
                slot_labels    = batch['slot_labels'].to(device)

                output     = self.bert(input_ids, attention_mask, token_type_ids)
                token_vecs = self.dropout(output.last_hidden_state)
                logits     = self.classifier(token_vecs)
                preds      = logits.argmax(dim=-1).cpu()

                # Only keep positions where true label != IGNORE_INDEX
                for pred_seq, true_seq in zip(preds, slot_labels.cpu()):
                    pred_labels, true_labels = [], []
                    for p, t in zip(pred_seq, true_seq):
                        if t.item() != IGNORE_INDEX:
                            pred_labels.append(id2slot[p.item()])
                            true_labels.append(id2slot[t.item()])
                    all_preds.append(pred_labels)
                    all_true.append(true_labels)

        if evaluate:
            from seqeval.metrics import f1_score
            return f1_score(all_true, all_preds)
        return all_preds


slot_model = sequence_labeling_model(num_slots=NUM_SLOTS)

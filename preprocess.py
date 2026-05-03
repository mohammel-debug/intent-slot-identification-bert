from transformers import BertTokenizerFast

tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

# TOKENISE
enc = tokenizer(
    words,
    is_split_into_words=True,  # tells BERT these are already split into words
    return_tensors='pt'
)


# LABEL ALIGNMENT


IGNORE_INDEX = -100

def align_labels(words, slots, enc):
    word_ids = enc.word_ids(batch_index=0)
    aligned = []
    prev_word_id = None

    for wid in word_ids:
        if wid is None:
            aligned.append(IGNORE_INDEX)        # CLS, SEP, PAD
        elif wid != prev_word_id:
            aligned.append(slots[wid])          # first subtoken → real label
        else:
            aligned.append(IGNORE_INDEX)        # continuation → ignore
        prev_word_id = wid
    return aligned

#LABEL ENCODING

# Intent label vocabulary 
intent_labels = sorted(set(train_intents + dev_intents + test_intents))
intent2id = {l: i for i, l in enumerate(intent_labels)}
id2intent  = {i: l for l, i in intent2id.items()}

# Slot label vocabulary
slot_labels = ['PAD'] + sorted(set(lbl for seq in train_slots + dev_slots + test_slots for lbl in seq))
slot2id = {l: i for i, l in enumerate(slot_labels)}
id2slot  = {i: l for l, i in slot2id.items()}

# PIPELINE
import torch
from torch.utils.data import Dataset, DataLoader

MAX_LEN = 64

class ATISDataset(Dataset):

    def __init__(self, texts, slots, intents):
        self.samples = []

        for words, slot_seq, intent in zip(texts, slots, intents):
            words = list(words)

            # 1. Tokenise with BERT
            enc = tokenizer(
                words,
                is_split_into_words=True,
                max_length=MAX_LEN,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )

            # 2. Align slot labels (your -100 function)
            word_ids = enc.word_ids(batch_index=0)
            prev_word_id = None
            label_ids = []
            for wid in word_ids:
                if wid is None:
                    label_ids.append(IGNORE_INDEX)       # CLS, SEP, PAD
                elif wid != prev_word_id:
                    label_ids.append(slot2id[slot_seq[wid]])  # first subtoken
                else:
                    label_ids.append(IGNORE_INDEX)       # continuation ##
                prev_word_id = wid

            # 3. Store as tensors
            self.samples.append({
                'input_ids'      : enc['input_ids'].squeeze(0),
                'attention_mask' : enc['attention_mask'].squeeze(0),
                'token_type_ids' : enc['token_type_ids'].squeeze(0),
                'slot_labels'    : torch.tensor(label_ids, dtype=torch.long),
                'intent_label'   : torch.tensor(intent2id[intent], dtype=torch.long),
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# Build all three datasets
train_dataset = ATISDataset(train_texts, train_slots, train_intents)
dev_dataset   = ATISDataset(dev_texts,   dev_slots,   dev_intents)
test_dataset  = ATISDataset(test_texts,  test_slots,  test_intents)

print(f"Train: {len(train_dataset)} | Dev: {len(dev_dataset)} | Test: {len(test_dataset)}")

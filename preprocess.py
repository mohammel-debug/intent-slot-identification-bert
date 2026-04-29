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

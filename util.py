import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import re
from transformers import DistilBertTokenizerFast
from torch.utils.data import Dataset
import pandas as pd


def get_tokenizer():
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    tokenizer.add_special_tokens({"additional_special_tokens": ["[E1]", "[E2]", "[E3]", "[E4]", "[E5]"]})
    return tokenizer

# Helper functions to detect tickers and mask them

def mask_tickers(text, max_tickers=5):
    """
    Mask tickers in text with the following rules:
    - Start with $
    - 1 to 4 letters, optional single dot (not first/last), only uppercase letters
    - Only first `max_tickers` tickers are masked; others remain unchanged
    Returns:
        masked_text, mapping
    """
    pattern = r"\$[A-Z\.]{1,5}"  # preliminary match, we'll filter invalids
    mapping = {}
    ticker_to_tag = {}
    next_idx = 1
    masked_count = 0

    def replace_func(match):
        nonlocal next_idx, masked_count
        if masked_count >= max_tickers:
            return match.group(0)  # leave remaining tickers unchanged

        ticker = match.group(0)[1:]  # remove $

        # Validation
        if len(ticker) > 4:
            return match.group(0)
        if ticker.count('.') > 1:
            return match.group(0)
        if ticker.startswith('.') or ticker.endswith('.'):
            return match.group(0)
        if not re.fullmatch(r"[A-Z]+\.?[A-Z]*", ticker):
            return match.group(0)

        # Assign placeholder
        if ticker not in ticker_to_tag:
            tag = f"[E{next_idx}]"
            ticker_to_tag[ticker] = tag
            mapping[tag] = ticker
            next_idx += 1
        else:
            tag = ticker_to_tag[ticker]

        masked_count += 1
        return tag

    masked_text = re.sub(pattern, replace_func, text)
    return masked_text, mapping

def batched_masking(tweet_batch):
    """
    Applies mask_tickers() to a DataLoader-style batch from Tweets1.
    Returns masked text and mappings as lists of len = batch_size
    """
    masked_texts, mappings = [], []

    for text in tweet_batch:
        masked, mapping = mask_tickers(text)
        masked_texts.append(masked)
        mappings.append(mapping)

    return masked_texts, mappings

# Helper functions to get the idx positions of the tokenized masked input text 
# and handle the forward call with optional per-entity argument

def get_entity_positions(tokenizer, tokens, mappings, max_pos=64):
    """
    Returns entity token index positions for each example in the batch,
    but skips any token indices >= max_pos.

    Args:
        tokenizer: tokenizer (used to convert tag -> id)
        tokens: tensor of shape [batch_size, seq_len] (torch.Tensor)
        mappings: list of dicts (one dict per example) with tags as keys
        max_pos: int, exclusive upper bound on allowed token positions
                 (positions >= max_pos will be ignored)
    Returns:
        entity_positions: List[List[List[int]]] where entity_positions[i]
                          is a list of entities for example i and each
                          entity is a list of token indices (< max_pos).
    """
    entity_positions = []
    for token_row, mapping in zip(tokens, mappings):
        positions = []
        for tag in mapping.keys():
            tag_id = tokenizer.convert_tokens_to_ids(tag)
            # find all positions where token == tag_id
            idxs = (token_row == tag_id).nonzero(as_tuple=True)[0]
            if idxs.numel() == 0:
                continue
            # keep only indices strictly less than max_pos
            filtered = idxs[idxs < max_pos]
            if filtered.numel() == 0:
                continue
            positions.append(filtered.tolist())
        entity_positions.append(positions)
    return entity_positions


def forward(model, tokenizer , batch , tokenizer_kwargs , per_entity_sentiment = False):
    text, mappings = batched_masking(batch["tweet"])
    tokenized = tokenizer(text, **tokenizer_kwargs)
    tokens = tokenized['input_ids']                 # [batch_size, seq_len]
    attention_mask = tokenized['attention_mask']    # [batch_size, seq_len], 1=real token, 0=pad

    # move to model device
    device = next(model.parameters()).device
    tokens = tokens.to(device)
    attention_mask = attention_mask.to(device)

    if per_entity_sentiment:
        entity_positions = get_entity_positions(tokenizer, tokens, mappings)
        out = model(tokens, entity_positions ,attention_mask)
    else: out = model(tokens, attention_mask= attention_mask)

    return out


class Teacher():
    """
    This class handles reading from disk/downloading of the FinTwitBERT-sentiment model.
    The class is callable on input text, using its own tokenizer for modular logit calculation.
    The logit outputs index order is:  0: neutral, 1: bullish, 2: bearish

    Ex Use:
    from util import Teacher
    teacher = Teacher()
    text = "Nice 9% pre market move for $para, pump my calls Uncle Buffett 🤑"
    teacher(text)
    """
    def __init__(self, tok_max_length):
        self.model_name = "StephanAkkerman/FinTwitBERT-sentiment"
        self.save_dir = "./teacher_model"
        self.teacher, self.tokenizer = self.get_teacher()
        self.max_length = tok_max_length

    def get_teacher(self):
        # Check if model is already saved locally
        if os.path.exists(self.save_dir) and os.path.isdir(self.save_dir):
            print("Loading model and tokenizer from disk...")
            t_tokenizer = AutoTokenizer.from_pretrained(self.save_dir)
            teacher = AutoModelForSequenceClassification.from_pretrained(self.save_dir)
        else:
            print("Downloading model and tokenizer from Hugging Face...")
            t_tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            teacher = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            # Save for future use
            os.makedirs(self.save_dir, exist_ok=True)
            teacher.save_pretrained(self.save_dir)
            t_tokenizer.save_pretrained(self.save_dir)

        teacher.eval() 
        return teacher, t_tokenizer

    @torch.no_grad()
    def __call__(self, text, device = None):
        model_device = next(self.teacher.parameters()).device
        device = torch.device(device) if device is not None else model_device
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length = self.max_length)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        out = self.teacher(**inputs)
        return out.logits
    
#-----------------------------------------------------------------------------------------------------

# Dataset of Financial Tweets for Distillation Training
class Tweets1(Dataset):
    """
    In this dataset we are given soft labels for tweets.
    The labels are also given as strings directly, rather than integers
    """
    def __init__(self, max_length=128, split="train", cache_dir="./Datasets"):
        self.max_length = max_length
        self.cache_dir = cache_dir

        self.label_order = ["Neutral", "Bullish", "Bearish"]
        self.label2id = {l: i for i, l in enumerate(self.label_order)}

        csv_path = os.path.join(cache_dir, f"financial_tweets_{split}.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Dataset not found at {csv_path}")

        self.data = pd.read_csv(csv_path)
        self.data = self.data.dropna(subset=["sentiment", "description"]).reset_index(drop=True)

    def _parse_soft_label(self, label_str):
        """
        Converts label like 'Bullish (60%)' into a soft probability vector.
        Order: [Neutral, Bullish, Bearish] (Same ordering as the teacher model FintwitBERT-Sentiment)
        # Example: 'Bullish (60%)' -> [0.4, 0.6, 0.0] (Neutral=0.4, Bullish=0.6, Bearish=0.0)
        Rules:
          - 'Bullish (60%)'  -> [0.4, 0.6, 0.0]
          - 'Neutral (50%)'  -> [0.5, 0.25, 0.25]
          - 'Bearish (80%)'  -> [0.2, 0.0, 0.8]
          - 'Bullish'        -> [1.0, 0.0, 0.0]
        """
        base_probs = torch.zeros(3, dtype=torch.float32)
        match = re.match(r"(Bullish|Bearish|Neutral)\s*\(?(\d{1,3}(?:\.\d+)?)?%?\)?", str(label_str), re.IGNORECASE)
        if not match:
            return base_probs  # fallback to all zeros if unknown

        sentiment = match.group(1).capitalize()
        conf = float(match.group(2)) / 100.0 if match.group(2) else 1.0

        if sentiment == "Bullish":
            base_probs[self.label2id["Bullish"]] = conf
            base_probs[self.label2id["Neutral"]] = 1.0 - conf
        elif sentiment == "Bearish":
            base_probs[self.label2id["Bearish"]] = conf
            base_probs[self.label2id["Neutral"]] = 1.0 - conf
        elif sentiment == "Neutral":
            base_probs[self.label2id["Neutral"]] = conf
            remain = 1.0 - conf
            base_probs[self.label2id["Bullish"]] = remain / 2
            base_probs[self.label2id["Bearish"]] = remain / 2

        return base_probs

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tweet = self.data.iloc[idx]["description"]
        label_str = self.data.iloc[idx]["sentiment"]
        soft_label = self._parse_soft_label(label_str)

        return {"tweet": tweet, "label": soft_label}


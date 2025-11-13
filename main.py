from util import Tweets1, Teacher, get_tokenizer
from train import distillation_train
from torch.utils.data import DataLoader
from model import EntitySentimentTransformer
import torch

def main():
    seed = 1337
    torch.manual_seed(seed)                
    torch.cuda.manual_seed_all(seed)

    # dataset and dataloader
    train_ds = Tweets1(split="train", cache_dir="./Datasets")
    val_ds   = Tweets1(split="val", cache_dir="./Datasets")

    train_dl = DataLoader(train_ds, batch_size= 32, shuffle = True)
    val_dl = DataLoader(val_ds, batch_size= 32, shuffle = True)

    # teacher & tokenizers
    teacher = Teacher(64) # 64 = max_len of tokenizer
    tokenizer = get_tokenizer()

    # config for a very small prototype & its tokenizer
    smallest_config = {"max_length": 64,
                    "embed_dim": 128,
                    "num_heads": 4,
                    "num_layers": 4,
                    "ffn_dim": 2 * 128,
                    "dropout": 0.1,
                    "vocab_size" : len(tokenizer),
                    "pad_token_id" : tokenizer.pad_token_id}

    tokenizer_kwargs = {"padding" : True, "truncation" : True, "max_length" : smallest_config['max_length'],
                        "return_tensors" : "pt", "return_attention_mask" : True }

    model_small = EntitySentimentTransformer(**smallest_config)

    print("Starting Training")
    # distillation train the model
    distillation_train(
            model_small , teacher, tokenizer, tokenizer_kwargs, train_dl, val_dl,
            n_epochs = 2, grad_accum_steps=1, lr=1e-6, temperature=2.0,
            eval_every=500,
            checkpoint_dir="./distill_checkpoints",
            device="cuda" if torch.cuda.is_available() else "cpu")
    print("Training Completed!")

if __name__ == "__main__":
    main()

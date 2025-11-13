import os
from tqdm import tqdm
import torch.nn.functional as F
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
import torch
from util import forward

def get_lambda(step_pct):
    """Helper function to calculate lambda to be used in distillation schedueling"""
    lambda_distill_start = 0.9
    lambda_distill_end = 0.3
    diff = lambda_distill_end - lambda_distill_start
    lambda_distill = lambda_distill_start + step_pct * diff
    lambda_supervised = 1 - lambda_distill
    return lambda_distill, lambda_supervised

@torch.no_grad()
def evaluate_distil(model,tokenizer, teacher, tokenizer_kwargs, dataloader, device,
             lambda_distill, lambda_supervised, temperature=2.0):
    """
    Evaluate the student model on a validation set.
    Computes both distillation (KD) and supervised (KL) losses.
    """
    sup_loss_fn = nn.KLDivLoss(reduction="batchmean")
    kl_loss_fn = nn.KLDivLoss(reduction="batchmean")

    total_sup_loss, total_dist_loss, total_loss = 0.0, 0.0, 0.0

    was_training = model.training
    model.eval()

    for batch in tqdm(dataloader, desc="Validation", leave=False):
        tweets, labels = batch["tweet"], batch["label"].to(device)
        # student & teacher outputs
        out = forward(model ,tokenizer, batch, tokenizer_kwargs, per_entity_sentiment = False)
        logits = out["pooled_logits"]
        logprobs_temp = F.log_softmax(logits / temperature, dim=-1)  # for KD term
        logprobs = F.log_softmax(logits, dim=-1)                     # for supervised KL term

        t_logits = teacher(tweets)
        soft_t_logits = F.softmax(t_logits / temperature, dim=-1)

        # Loss
        kd_loss = kl_loss_fn(logprobs_temp, soft_t_logits) * (temperature ** 2)
        sup_loss = sup_loss_fn(logprobs, labels)
        loss = lambda_distill * kd_loss + lambda_supervised * sup_loss

        total_sup_loss += sup_loss.item()
        total_dist_loss += kd_loss.item()
        total_loss += loss.item()

    total_sup_loss /= len(dataloader)
    total_dist_loss /= len(dataloader)
    total_loss /= len(dataloader)

    print(f"Evaluation | KD Loss = {total_dist_loss:.4f} | Sup Loss = {total_sup_loss:.4f} | Total Loss = {total_loss:.4f}")

    if was_training:
        model.train()

def distillation_train(
        model, teacher, tokenizer, tokenizer_kwargs, train_loader, val_loader,
        n_epochs = 1, grad_accum_steps=2, lr=1e-4, temperature=2.0, eval_every=500,
        checkpoint_dir="./distill_checkpoints",
        device="cuda" if torch.cuda.is_available() else "cpu"):
    
    # Setup
    sup_loss_fn = nn.KLDivLoss(reduction="batchmean")
    kl_loss_fn = nn.KLDivLoss(reduction="batchmean")
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    # directory for saving checkpoints
    os.makedirs(checkpoint_dir, exist_ok=True)

    # number of warmup steps = 5% of total
    n_batches = len(train_loader)
    warmup_steps = int(0.05 * n_batches * n_epochs)
    total_steps = n_batches * n_epochs
    cosine_t_max = max(1, total_steps - warmup_steps)

    # linear warmup, then cosine decay
    scheduler = SequentialLR(
    optimizer,
    schedulers=[
        LinearLR(optimizer, start_factor=0.1, total_iters = warmup_steps),
        CosineAnnealingLR(optimizer, T_max = cosine_t_max)
    ],
    milestones=[warmup_steps])

    device = torch.device(device)
    model.to(device)
    teacher.teacher.to(device)
    model.train()

    # Training loop
    for epoch in range(1, n_epochs + 1):
        print(f"Epoch {epoch} / {n_epochs}")

        for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)):
            tweets, labels = batch["tweet"] , batch["label"].to(device) # label is (B, 3) tensor

            # get teacher logits for the CLS token, tokenization is handled by the Teacher class
            with torch.no_grad():
                t_logits = teacher(tweets)
                soft_t_logits = F.softmax(t_logits / temperature, dim=-1)

            # Preprocessing: ensure tensors are on the same device
            out = forward(model ,tokenizer, batch, tokenizer_kwargs, per_entity_sentiment = False)
            logits = out["pooled_logits"]
            logprobs_temp = F.log_softmax(logits / temperature, dim=-1)  # for KD term
            logprobs = F.log_softmax(logits, dim=-1)                     # for supervised KL term

            kd_loss = kl_loss_fn(logprobs_temp, soft_t_logits) * (temperature ** 2)
            sup_loss = sup_loss_fn(logprobs, labels)
            
            # calculate the lambda for the mixed loss
            global_step = (epoch - 1) * len(train_loader) + step
            step_pct = global_step / total_steps
            lambda_distill, lambda_supervised = get_lambda(step_pct)

            # Loss & Backwardpass
            loss = lambda_distill * kd_loss + lambda_supervised * sup_loss
            loss = loss / grad_accum_steps
            loss.backward()
            if (step + 1) % grad_accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

            if (step + 1) % (100 * grad_accum_steps) == 0:
                print(f"Iteration {(step +1) // grad_accum_steps} |  Loss = {(loss.item() * grad_accum_steps):.4f}")
            
            if (step + 1) % (eval_every * grad_accum_steps) == 0:
                evaluate_distil(model,tokenizer, teacher, tokenizer_kwargs, val_loader, device,
            lambda_distill, lambda_supervised, temperature=2.0)
        
        ckpt_path = os.path.join(checkpoint_dir, f"small_model{epoch}.pt")
        torch.save({
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "epoch": epoch
        }, ckpt_path)

        print(f"Saved checkpoint to {ckpt_path}")
        # later load with: model.load_state_dict(torch.load("./distill_checkpoints/small_model3.pt"))


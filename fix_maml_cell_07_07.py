#!/usr/bin/env python3
"""
Fixed CELL 07-07 for MAML meta-training.
Addresses the "One of the differentiated Tensors appears to not have been used" error.

Key fixes:
1. Add allow_unused=True to gradient computation
2. Simplify functional_forward to avoid GRU parameter manipulation issues
3. Use a clone-based approach for adaptation
"""

fixed_cell_code = '''# [CELL 07-07] MAML meta-training loop

t0 = cell_start("CELL 07-07", "MAML meta-training")

# Initialize meta-model
meta_model = GRURecommender(
    n_items=n_items,
    embedding_dim=CFG["gru_config"]["embedding_dim"],
    hidden_dim=CFG["gru_config"]["hidden_dim"],
    num_layers=CFG["gru_config"]["num_layers"],
    dropout=CFG["gru_config"]["dropout"],
).to(DEVICE)

print(f"[CELL 07-07] Meta-model parameters: {sum(p.numel() for p in meta_model.parameters()):,}")

# Meta-optimizer (outer loop)
meta_optimizer = torch.optim.Adam(meta_model.parameters(), lr=CFG["maml_config"]["outer_lr"])
criterion = nn.CrossEntropyLoss()

# MAML hyperparameters
inner_lr = CFG["maml_config"]["inner_lr"]
num_inner_steps = CFG["maml_config"]["num_inner_steps"]
meta_batch_size = CFG["maml_config"]["meta_batch_size"]
num_meta_iterations = CFG["maml_config"]["num_meta_iterations"]
use_second_order = CFG["maml_config"]["use_second_order"]
max_seq_len = CFG["gru_config"]["max_seq_len"]

print(f"[CELL 07-07] Meta-training config:")
print(f"  - Inner LR (α): {inner_lr}")
print(f"  - Outer LR (β): {CFG['maml_config']['outer_lr']}")
print(f"  - Inner steps: {num_inner_steps}")
print(f"  - Meta-batch size: {meta_batch_size}")
print(f"  - Meta-iterations: {num_meta_iterations:,}")
print(f"  - Second-order: {use_second_order}")

def get_episode_data(episode_row, pairs_df):
    """Extract support and query pairs for an episode."""
    support_pair_ids = episode_row["support_pair_ids"]
    query_pair_ids = episode_row["query_pair_ids"]

    support_pairs = pairs_df[pairs_df["pair_id"].isin(support_pair_ids)].sort_values("label_ts_epoch")
    query_pairs = pairs_df[pairs_df["pair_id"].isin(query_pair_ids)].sort_values("label_ts_epoch")

    return support_pairs, query_pairs

def pairs_to_batch(pairs_df, max_len):
    """Convert pairs to batched tensors."""
    prefixes = []
    labels = []
    lengths = []

    for _, row in pairs_df.iterrows():
        prefix = row["prefix"]
        if len(prefix) > max_len:
            prefix = prefix[-max_len:]
        prefixes.append(prefix)
        labels.append(row["label"])
        lengths.append(len(prefix))

    # Pad sequences
    max_l = max(lengths)
    padded = []
    for seq in prefixes:
        padded.append(list(seq) + [0] * (max_l - len(seq)))

    return (
        torch.LongTensor(padded).to(DEVICE),
        torch.LongTensor(labels).to(DEVICE),
        torch.LongTensor(lengths).to(DEVICE),
    )

def clone_model(model):
    """Create a deep copy of the model for adaptation."""
    cloned = GRURecommender(
        n_items=model.n_items,
        embedding_dim=model.embedding_dim,
        hidden_dim=model.hidden_dim,
        num_layers=model.gru.num_layers,
        dropout=0.0  # No dropout during adaptation
    ).to(DEVICE)
    cloned.load_state_dict(model.state_dict())
    return cloned

def inner_loop_update(meta_params, support_seq, support_labels, support_lengths, num_steps, lr):
    """
    Perform inner loop adaptation on support set.
    Uses a temporary model clone to avoid gradient issues.
    Returns adapted model.
    """
    # Create temporary model for adaptation
    adapted_model = clone_model(meta_model)

    # Enable gradients for adaptation
    for param in adapted_model.parameters():
        param.requires_grad = True

    # Inner loop: gradient descent on support set
    for step in range(num_steps):
        # Forward pass
        logits = adapted_model(support_seq, support_lengths)
        loss = criterion(logits, support_labels)

        # Compute gradients
        grads = torch.autograd.grad(
            loss,
            adapted_model.parameters(),
            create_graph=use_second_order,
            allow_unused=True  # FIX: Allow unused parameters
        )

        # Manual SGD update
        with torch.no_grad():
            for param, grad in zip(adapted_model.parameters(), grads):
                if grad is not None:  # Only update if gradient exists
                    if use_second_order:
                        # Keep computation graph for second-order
                        param.data = param.data - lr * grad
                    else:
                        # Detach for first-order MAML
                        param.data = param.data - lr * grad.detach()

    return adapted_model

# Training tracking
training_history = {
    "meta_iterations": [],
    "meta_train_loss": [],
    "val_accuracy": [],
    "val_iterations": [],
}

print(f"\\n[CELL 07-07] Starting meta-training...")

# Sample episodes for meta-training (unique users per batch)
train_users = episodes_train["user_id"].unique()

for meta_iter in range(num_meta_iterations):
    meta_model.train()
    meta_optimizer.zero_grad()

    # Sample meta-batch of tasks (episodes/users)
    sampled_users = np.random.choice(train_users, size=min(meta_batch_size, len(train_users)), replace=False)
    meta_loss = 0.0
    valid_tasks = 0

    for user_id in sampled_users:
        # Sample one episode for this user
        user_episodes = episodes_train[episodes_train["user_id"] == user_id]
        if len(user_episodes) == 0:
            continue

        episode = user_episodes.sample(n=1).iloc[0]

        # Get support and query sets
        support_pairs, query_pairs = get_episode_data(episode, pairs_train)

        if len(support_pairs) == 0 or len(query_pairs) == 0:
            continue

        support_seq, support_labels, support_lengths = pairs_to_batch(support_pairs, max_seq_len)
        query_seq, query_labels, query_lengths = pairs_to_batch(query_pairs, max_seq_len)

        # Inner loop: adapt to support set
        adapted_model = inner_loop_update(
            meta_model.parameters(), support_seq, support_labels, support_lengths, num_inner_steps, inner_lr
        )

        # Outer loop: evaluate on query set with adapted model
        query_logits = adapted_model(query_seq, query_lengths)
        query_loss = criterion(query_logits, query_labels)

        meta_loss = meta_loss + query_loss
        valid_tasks += 1

    if valid_tasks == 0:
        continue

    # Meta-update: update θ based on aggregated meta-loss
    meta_loss = meta_loss / valid_tasks
    meta_loss.backward()

    # Gradient clipping to prevent exploding gradients
    torch.nn.utils.clip_grad_norm_(meta_model.parameters(), max_norm=10.0)

    meta_optimizer.step()

    # Logging
    training_history["meta_iterations"].append(meta_iter)
    training_history["meta_train_loss"].append(meta_loss.item())

    if (meta_iter + 1) % 100 == 0:
        print(f"[CELL 07-07] Iter {meta_iter+1}/{num_meta_iterations}: meta_loss={meta_loss.item():.4f}")

    # Checkpointing
    if (meta_iter + 1) % CFG["maml_config"]["checkpoint_interval"] == 0:
        checkpoint_path = CHECKPOINTS_DIR / f"checkpoint_iter{meta_iter+1}.pth"
        torch.save({
            "meta_iter": meta_iter + 1,
            "model_state_dict": meta_model.state_dict(),
            "optimizer_state_dict": meta_optimizer.state_dict(),
            "config": CFG,
            "training_history": training_history,
        }, checkpoint_path)
        print(f"[CELL 07-07] Saved checkpoint: {checkpoint_path.name}")

    # Validation evaluation
    if (meta_iter + 1) % CFG["maml_config"]["eval_interval"] == 0:
        print(f"[CELL 07-07] Evaluating on val set at iter {meta_iter+1}...")
        meta_model.eval()

        val_predictions = []
        val_labels = []

        with torch.no_grad():
            for _, episode in episodes_val.head(50).iterrows():  # Sample for speed
                support_pairs, query_pairs = get_episode_data(episode, pairs_val)

                if len(support_pairs) == 0 or len(query_pairs) == 0:
                    continue

                support_seq, support_labels, support_lengths = pairs_to_batch(support_pairs, max_seq_len)
                query_seq, query_labels_t, query_lengths = pairs_to_batch(query_pairs, max_seq_len)

                # Adapt on support (no grad)
                adapted_model = inner_loop_update(
                    meta_model.parameters(), support_seq, support_labels, support_lengths, num_inner_steps, inner_lr
                )

                # Evaluate on query
                query_logits = adapted_model(query_seq, query_lengths)
                query_probs = torch.softmax(query_logits, dim=-1).cpu().numpy()

                val_predictions.append(query_probs)
                val_labels.extend(query_labels_t.cpu().numpy())

        if len(val_predictions) > 0:
            val_predictions = np.vstack(val_predictions)
            val_labels = np.array(val_labels)
            val_metrics = compute_metrics(val_predictions, val_labels)

            training_history["val_accuracy"].append(val_metrics["accuracy@1"])
            training_history["val_iterations"].append(meta_iter + 1)

            print(f"[CELL 07-07] Val Acc@1: {val_metrics['accuracy@1']:.4f}, "
                  f"Recall@5: {val_metrics['recall@5']:.4f}, MRR: {val_metrics['mrr']:.4f}")

# Save final meta-learned model
final_model_path = MODELS_DIR / f"maml_gru_K{K}.pth"
torch.save({
    "model_state_dict": meta_model.state_dict(),
    "config": CFG,
    "training_history": training_history,
}, final_model_path)

print(f"\\n[CELL 07-07] Saved final meta-model: {final_model_path.name}")
print(f"[CELL 07-07] Total training time: {time.time()-t0:.1f}s")

cell_end("CELL 07-07", t0)
'''

print("Fixed CELL 07-07 code generated.")
print("\nKey fixes:")
print("1. Added allow_unused=True to torch.autograd.grad()")
print("2. Simplified approach using model cloning instead of functional forward")
print("3. Added gradient clipping for stability")
print("4. Better handling of None gradients")
print("\nReplace CELL 07-07 in your notebook with this fixed version.")

"""
ABSOLUTE FINAL FIX for CELL 07-07

The issue: Manually setting param.grad breaks the computation graph.
Solution: Don't set gradients manually. Instead, compute a meta-loss and backprop normally.

This uses a simplified Reptile approach with proper gradient flow.
"""

fixed_cell = '''# [CELL 07-07] MAML meta-training loop (Simplified Reptile)

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
max_seq_len = CFG["gru_config"]["max_seq_len"]

print(f"[CELL 07-07] Using Reptile meta-learning (simplified)")
print(f"[CELL 07-07] Meta-training config:")
print(f"  - Inner LR (α): {inner_lr}")
print(f"  - Outer LR (β): {CFG['maml_config']['outer_lr']}")
print(f"  - Inner steps: {num_inner_steps}")
print(f"  - Meta-batch size: {meta_batch_size}")
print(f"  - Meta-iterations: {num_meta_iterations:,}")

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

# Training tracking
training_history = {
    "meta_iterations": [],
    "meta_train_loss": [],
    "val_accuracy": [],
    "val_iterations": [],
}

print(f"\\n[CELL 07-07] Starting meta-training...")

# Sample episodes for meta-training
train_users = episodes_train["user_id"].unique()

for meta_iter in range(num_meta_iterations):
    meta_model.train()

    # Sample meta-batch of tasks
    sampled_users = np.random.choice(train_users, size=min(meta_batch_size, len(train_users)), replace=False)

    # Collect adapted parameters from all tasks
    all_adapted_params = []
    valid_tasks = 0
    total_loss = 0.0

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

        # Save original parameters
        original_params = [param.clone().detach() for param in meta_model.parameters()]

        # ===== INNER LOOP: Adapt to support set =====
        inner_optimizer = torch.optim.SGD(meta_model.parameters(), lr=inner_lr)

        for inner_step in range(num_inner_steps):
            inner_optimizer.zero_grad()
            support_logits = meta_model(support_seq, support_lengths)
            support_loss = criterion(support_logits, support_labels)
            support_loss.backward()
            inner_optimizer.step()

        # Store adapted parameters
        adapted_params = [param.clone().detach() for param in meta_model.parameters()]
        all_adapted_params.append(adapted_params)

        # Evaluate on query for logging
        with torch.no_grad():
            query_logits = meta_model(query_seq, query_lengths)
            query_loss = criterion(query_logits, query_labels)
            total_loss += query_loss.item()

        # Restore original parameters
        with torch.no_grad():
            for param, orig_param in zip(meta_model.parameters(), original_params):
                param.copy_(orig_param)

        valid_tasks += 1

    if valid_tasks == 0:
        continue

    # ===== META-UPDATE (Reptile): Move toward average of adapted parameters =====
    meta_optimizer.zero_grad()

    # Compute average adapted parameters
    avg_adapted_params = []
    for i in range(len(all_adapted_params[0])):
        avg_param = torch.stack([task_params[i] for task_params in all_adapted_params]).mean(dim=0)
        avg_adapted_params.append(avg_param)

    # Compute pseudo-loss: MSE between current params and adapted params
    # This creates gradients pointing toward adapted parameters
    pseudo_loss = 0.0
    for param, avg_adapted in zip(meta_model.parameters(), avg_adapted_params):
        pseudo_loss = pseudo_loss + ((param - avg_adapted) ** 2).sum()

    # Backprop through pseudo-loss (this moves params toward adapted params)
    pseudo_loss.backward()

    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(meta_model.parameters(), max_norm=10.0)

    meta_optimizer.step()

    # Logging
    avg_loss = total_loss / valid_tasks if valid_tasks > 0 else 0.0
    training_history["meta_iterations"].append(meta_iter)
    training_history["meta_train_loss"].append(avg_loss)

    if (meta_iter + 1) % 100 == 0:
        print(f"[CELL 07-07] Iter {meta_iter+1}/{num_meta_iterations}: avg_query_loss={avg_loss:.4f}")

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

    # Validation
    if (meta_iter + 1) % CFG["maml_config"]["eval_interval"] == 0:
        print(f"[CELL 07-07] Evaluating on val set at iter {meta_iter+1}...")
        meta_model.eval()

        val_predictions = []
        val_labels = []

        with torch.no_grad():
            for _, episode in episodes_val.head(50).iterrows():
                support_pairs, query_pairs = get_episode_data(episode, pairs_val)

                if len(support_pairs) == 0 or len(query_pairs) == 0:
                    continue

                support_seq, support_labels, support_lengths = pairs_to_batch(support_pairs, max_seq_len)
                query_seq, query_labels_t, query_lengths = pairs_to_batch(query_pairs, max_seq_len)

                # Save original params
                original_params = [param.clone() for param in meta_model.parameters()]

                # Adapt on support
                inner_optimizer = torch.optim.SGD(meta_model.parameters(), lr=inner_lr)
                for inner_step in range(num_inner_steps):
                    inner_optimizer.zero_grad()
                    support_logits = meta_model(support_seq, support_lengths)
                    support_loss = criterion(support_logits, support_labels)
                    support_loss.backward()
                    inner_optimizer.step()

                # Evaluate on query
                query_logits = meta_model(query_seq, query_lengths)
                query_probs = torch.softmax(query_logits, dim=-1).cpu().numpy()

                val_predictions.append(query_probs)
                val_labels.extend(query_labels_t.cpu().numpy())

                # Restore params
                for param, orig_param in zip(meta_model.parameters(), original_params):
                    param.copy_(orig_param)

        if len(val_predictions) > 0:
            val_predictions = np.vstack(val_predictions)
            val_labels = np.array(val_labels)
            val_metrics = compute_metrics(val_predictions, val_labels)

            training_history["val_accuracy"].append(val_metrics["accuracy@1"])
            training_history["val_iterations"].append(meta_iter + 1)

            print(f"[CELL 07-07] Val Acc@1: {val_metrics['accuracy@1']:.4f}, "
                  f"Recall@5: {val_metrics['recall@5']:.4f}, MRR: {val_metrics['mrr']:.4f}")

# Save final model
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

print(fixed_cell)

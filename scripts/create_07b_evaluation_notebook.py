"""
Create Notebook 07b: MAML Evaluation (Load Trained Model)

This script creates a separate evaluation notebook that:
1. Loads the trained MAML model from checkpoint
2. Contains only evaluation cells (CELL 07-08 through 07-14)
3. Fixes the meta_model.eval() issue in functional_forward cells
4. Adds missing CELL 07-08 (zero-shot evaluation)
"""

import json
from pathlib import Path

# Paths
REPO_ROOT = Path(__file__).parent.parent
NB_ORIG = REPO_ROOT / "notebooks" / "07_maml_xuetangx.ipynb"
NB_EVAL = REPO_ROOT / "notebooks" / "07b_maml_evaluation.ipynb"

print(f"Loading original notebook: {NB_ORIG}")
with open(NB_ORIG, encoding='utf-8') as f:
    nb_orig = json.load(f)

print(f"Total cells in original: {len(nb_orig['cells'])}")

# Create new notebook
nb_eval = {
    'cells': [],
    'metadata': nb_orig['metadata'],
    'nbformat': nb_orig['nbformat'],
    'nbformat_minor': nb_orig['nbformat_minor']
}

# ========== CELL 0: Title ==========
title_cell = {
    'cell_type': 'markdown',
    'metadata': {},
    'source': [
        '# Notebook 07b: MAML Evaluation (Load Trained Model)\n',
        '\n',
        '**Purpose**: Evaluate trained MAML model from Notebook 07\n',
        '\n',
        'This notebook:\n',
        '- Loads the meta-trained model checkpoint from Notebook 07\n',
        '- Runs evaluation cells only (CELL 07-08 through 07-14)\n',
        '- Does NOT re-run 24h training\n',
        '- Fixes `meta_model.eval()` issue in functional_forward cells\n',
        '\n',
        '**Use Case**: Re-run evaluations without re-training\n',
        '\n',
        '---\n'
    ]
}
nb_eval['cells'].append(title_cell)

# ========== Add setup cells 0-6 from original ==========
for i in range(7):  # cells 0-6
    nb_eval['cells'].append(nb_orig['cells'][i])

print(f"Added cells 0-6 (setup and data loading)")

# ========== NEW CELL: Load trained model ==========
load_model_cell = {
    'cell_type': 'code',
    'execution_count': None,
    'metadata': {},
    'outputs': [],
    'source': [
        '# [CELL 07-07b] Load trained MAML model from checkpoint\n',
        '\n',
        't0 = cell_start("CELL 07-07b", "Load trained model")\n',
        '\n',
        'print("[CELL 07-07b] Loading trained MAML model...")\n',
        '\n',
        '# Initialize model (same architecture as training)\n',
        'meta_model = GRURecommender(\n',
        '    n_items=n_items,\n',
        '    embedding_dim=CFG["gru_config"]["embedding_dim"],\n',
        '    hidden_dim=CFG["gru_config"]["hidden_dim"],\n',
        '    num_layers=CFG["gru_config"]["num_layers"],\n',
        '    dropout=CFG["gru_config"]["dropout"],\n',
        ').to(DEVICE)\n',
        '\n',
        '# Load checkpoint (try final model first, then latest checkpoint)\n',
        'final_model_path = MODELS_DIR / f"maml_gru_K{K}.pth"\n',
        'if final_model_path.exists():\n',
        '    print(f"[CELL 07-07b] Loading final model: {final_model_path.name}")\n',
        '    checkpoint = torch.load(final_model_path, map_location=DEVICE)\n',
        '    meta_model.load_state_dict(checkpoint["model_state_dict"])\n',
        '    if "training_history" in checkpoint:\n',
        '        training_history = checkpoint["training_history"]\n',
        '        print(f"[CELL 07-07b] Training history loaded (last iter: {training_history[\'meta_iterations\'][-1] if training_history[\'meta_iterations\'] else \'N/A\'})")\n',
        'else:\n',
        '    # Find latest checkpoint\n',
        '    checkpoints = sorted(CHECKPOINTS_DIR.glob("checkpoint_iter*.pth"))\n',
        '    if checkpoints:\n',
        '        latest_checkpoint = checkpoints[-1]\n',
        '        print(f"[CELL 07-07b] Loading checkpoint: {latest_checkpoint.name}")\n',
        '        checkpoint = torch.load(latest_checkpoint, map_location=DEVICE)\n',
        '        meta_model.load_state_dict(checkpoint["model_state_dict"])\n',
        '        if "training_history" in checkpoint:\n',
        '            training_history = checkpoint["training_history"]\n',
        '            print(f"[CELL 07-07b] Loaded from iteration {checkpoint[\'meta_iter\']}")\n',
        '    else:\n',
        '        raise FileNotFoundError(f"No trained model found in {MODELS_DIR} or {CHECKPOINTS_DIR}")\n',
        '\n',
        'print(f"[CELL 07-07b] Model loaded successfully")\n',
        'print(f"[CELL 07-07b] Model parameters: {sum(p.numel() for p in meta_model.parameters()):,}")\n',
        '\n',
        '# Copy MAML hyperparameters from config (needed for evaluation)\n',
        'inner_lr = CFG["maml_config"]["inner_lr"]\n',
        'num_inner_steps = CFG["maml_config"]["num_inner_steps"]\n',
        'max_seq_len = CFG["gru_config"]["max_seq_len"]\n',
        'criterion = nn.CrossEntropyLoss()\n',
        '\n',
        'print(f"[CELL 07-07b] Evaluation config:")\n',
        'print(f"  - Inner LR (α): {inner_lr}")\n',
        'print(f"  - Inner steps: {num_inner_steps}")\n',
        'print(f"  - Max seq length: {max_seq_len}")\n',
        '\n',
        '# Copy helper functions from training (needed for evaluation)\n',
        'def get_episode_data(episode_row, pairs_df):\n',
        '    """Extract support and query pairs for an episode."""\n',
        '    support_pair_ids = episode_row["support_pair_ids"]\n',
        '    query_pair_ids = episode_row["query_pair_ids"]\n',
        '\n',
        '    support_pairs = pairs_df[pairs_df["pair_id"].isin(support_pair_ids)].sort_values("label_ts_epoch")\n',
        '    query_pairs = pairs_df[pairs_df["pair_id"].isin(query_pair_ids)].sort_values("label_ts_epoch")\n',
        '\n',
        '    return support_pairs, query_pairs\n',
        '\n',
        'def pairs_to_batch(pairs_df, max_len):\n',
        '    """Convert pairs to batched tensors."""\n',
        '    prefixes = []\n',
        '    labels = []\n',
        '    lengths = []\n',
        '\n',
        '    for _, row in pairs_df.iterrows():\n',
        '        prefix = row["prefix"]\n',
        '        if len(prefix) > max_len:\n',
        '            prefix = prefix[-max_len:]\n',
        '        prefixes.append(prefix)\n',
        '        labels.append(row["label"])\n',
        '        lengths.append(len(prefix))\n',
        '\n',
        '    # Pad sequences\n',
        '    max_l = max(lengths)\n',
        '    padded = []\n',
        '    for seq in prefixes:\n',
        '        padded.append(list(seq) + [0] * (max_l - len(seq)))\n',
        '\n',
        '    return (\n',
        '        torch.LongTensor(padded).to(DEVICE),\n',
        '        torch.LongTensor(labels).to(DEVICE),\n',
        '        torch.LongTensor(lengths).to(DEVICE),\n',
        '    )\n',
        '\n',
        '# Functional forward pass (needed for evaluation)\n',
        'def functional_forward(seq, lengths, params, hidden_dim, n_items):\n',
        '    """Functional forward pass using explicit parameters."""\n',
        '    batch_size = seq.size(0)\n',
        '    \n',
        '    # 1. Embedding\n',
        '    emb = F.embedding(seq, params["embedding.weight"], padding_idx=0)\n',
        '    \n',
        '    # 2. GRU (manual implementation)\n',
        '    h = torch.zeros(batch_size, hidden_dim, device=seq.device)\n',
        '    w_ih = params["gru.weight_ih_l0"]\n',
        '    w_hh = params["gru.weight_hh_l0"]\n',
        '    b_ih = params["gru.bias_ih_l0"]\n',
        '    b_hh = params["gru.bias_hh_l0"]\n',
        '    \n',
        '    for t in range(emb.size(1)):\n',
        '        x_t = emb[:, t, :]\n',
        '        gi = F.linear(x_t, w_ih, b_ih)\n',
        '        gh = F.linear(h, w_hh, b_hh)\n',
        '        i_r, i_z, i_n = gi.chunk(3, 1)\n',
        '        h_r, h_z, h_n = gh.chunk(3, 1)\n',
        '        \n',
        '        r = torch.sigmoid(i_r + h_r)\n',
        '        z = torch.sigmoid(i_z + h_z)\n',
        '        n = torch.tanh(i_n + r * h_n)\n',
        '        h_new = (1 - z) * n + z * h\n',
        '        \n',
        '        mask = (lengths > t).unsqueeze(1).float()\n',
        '        h = mask * h_new + (1 - mask) * h\n',
        '    \n',
        '    # 3. FC layer\n',
        '    logits = F.linear(h, params["fc.weight"], params["fc.bias"])\n',
        '    return logits\n',
        '\n',
        'print(f"\\n[CELL 07-07b] Ready for evaluation!")\n',
        '\n',
        'cell_end("CELL 07-07b", t0)\n'
    ]
}
nb_eval['cells'].append(load_model_cell)

print("Added CELL 07-07b (load trained model)")

# ========== NEW CELL 07-08: Zero-shot evaluation ==========
cell_07_08 = {
    'cell_type': 'code',
    'execution_count': None,
    'metadata': {},
    'outputs': [],
    'source': [
        '# [CELL 07-08] Meta-testing: Zero-shot (K=0) - no adaptation\n',
        '\n',
        't0 = cell_start("CELL 07-08", "Zero-shot evaluation (K=0)")\n',
        '\n',
        'print("[CELL 07-08] Evaluating meta-learned model WITHOUT adaptation (zero-shot)...")\n',
        '\n',
        'meta_model.eval()\n',
        'zeroshot_predictions = []\n',
        'zeroshot_labels = []\n',
        '\n',
        'with torch.no_grad():  # Pure inference, no gradients needed\n',
        '    for _, episode in episodes_test.iterrows():\n',
        '        support_pairs, query_pairs = get_episode_data(episode, pairs_test)\n',
        '\n',
        '        if len(query_pairs) == 0:\n',
        '            continue\n',
        '\n',
        '        # Only use query set (no support set adaptation)\n',
        '        query_seq, query_labels_test, query_lengths = pairs_to_batch(query_pairs, max_seq_len)\n',
        '\n',
        '        # Use original meta-learned model (no adaptation)\n',
        '        query_logits = meta_model(query_seq, query_lengths)\n',
        '        query_probs = torch.softmax(query_logits, dim=-1).cpu().numpy()\n',
        '\n',
        '        zeroshot_predictions.append(query_probs)\n',
        '        zeroshot_labels.extend(query_labels_test.cpu().numpy())\n',
        '\n',
        '# Compute metrics\n',
        'if len(zeroshot_predictions) > 0:\n',
        '    zeroshot_predictions = np.vstack(zeroshot_predictions)\n',
        '    zeroshot_labels = np.array(zeroshot_labels)\n',
        '    zeroshot_metrics = compute_metrics(zeroshot_predictions, zeroshot_labels)\n',
        '\n',
        '    print(f"\\n[CELL 07-08] Zero-shot Results (No Adaptation):")\n',
        '    print(f"  Accuracy@1:  {zeroshot_metrics[\'accuracy@1\']:.4f}")\n',
        '    print(f"  Accuracy@3:  {zeroshot_metrics[\'accuracy@3\']:.4f}")\n',
        '    print(f"  Accuracy@5:  {zeroshot_metrics[\'accuracy@5\']:.4f}")\n',
        '    print(f"  Recall@5:    {zeroshot_metrics[\'recall@5\']:.4f}")\n',
        '    print(f"  Recall@10:   {zeroshot_metrics[\'recall@10\']:.4f}")\n',
        '    print(f"  MRR:         {zeroshot_metrics[\'mrr\']:.4f}")\n',
        'else:\n',
        '    print("[CELL 07-08] WARNING: No predictions generated")\n',
        '    zeroshot_metrics = {}\n',
        '\n',
        'cell_end("CELL 07-08", t0)\n'
    ]
}
nb_eval['cells'].append(cell_07_08)

print("Added CELL 07-08 (zero-shot evaluation) [NEW]")

# ========== Load and fix CELL 07-09 from original ==========
cell_07_09_orig = nb_orig['cells'][9]['source']
# Remove meta_model.eval() line
cell_07_09_fixed = []
for line in cell_07_09_orig:
    if 'meta_model.eval()' not in line or line.strip().startswith('#'):
        cell_07_09_fixed.append(line)
    else:
        # Replace with comment explaining why we don't use eval()
        cell_07_09_fixed.append('# NOTE: Do NOT call meta_model.eval() - functional_forward needs gradients\n')
        cell_07_09_fixed.append('# The cuDNN RNN backend requires training mode for backward pass\n')
        cell_07_09_fixed.append('\n')

cell_07_09 = {
    'cell_type': 'code',
    'execution_count': None,
    'metadata': {},
    'outputs': [],
    'source': cell_07_09_fixed
}
nb_eval['cells'].append(cell_07_09)

print("Added CELL 07-09 (few-shot K=5) - FIXED: removed meta_model.eval()")

# ========== Add remaining cells 10-16 from original ==========
for i in range(10, len(nb_orig['cells'])):
    cell = nb_orig['cells'][i].copy()

    # Fix meta_model.eval() in code cells if present
    if cell['cell_type'] == 'code' and 'source' in cell:
        source_fixed = []
        for line in cell['source']:
            if 'meta_model.eval()' in line and not line.strip().startswith('#'):
                source_fixed.append('# NOTE: Do NOT call meta_model.eval() - functional_forward needs gradients\n')
            else:
                source_fixed.append(line)
        cell['source'] = source_fixed

    nb_eval['cells'].append(cell)

print(f"Added cells 10-16 (ablations, analysis, reporting)")

# ========== Save new notebook ==========
print(f"\nTotal cells in evaluation notebook: {len(nb_eval['cells'])}")

with open(NB_EVAL, 'w', encoding='utf-8') as f:
    json.dump(nb_eval, f, indent=1, ensure_ascii=False)

print(f"\n[SUCCESS] Created: {NB_EVAL}")
print(f"\nNotebook 07b is ready to run!")
print(f"It will load the trained model and run all evaluation cells.")

"""
CORRECT FIX for CELL 07-07

The issue: `with torch.no_grad():` wraps the entire inner loop including backward().
Solution: Remove torch.no_grad() from around backward(), only use it for parameter updates.

This is the correct FOMAML implementation.
"""

# OLD CODE (BROKEN):
"""
        # ===== INNER LOOP: Adapt to support set (detached from meta-graph) =====
        with torch.no_grad():
            for inner_step in range(num_inner_steps):
                # Forward
                support_logits = meta_model(support_seq, support_lengths)
                support_loss = criterion(support_logits, support_labels)

                # Manual gradient computation and update
                meta_model.zero_grad()
                support_loss.backward()

                # SGD step
                for param in meta_model.parameters():
                    if param.grad is not None:
                        param.data = param.data - inner_lr * param.grad.data
"""

# NEW CODE (CORRECT):
"""
        # ===== INNER LOOP: Adapt to support set (FOMAML - detach gradients after computation) =====
        for inner_step in range(num_inner_steps):
            # Forward (WITH gradient tracking)
            support_logits = meta_model(support_seq, support_lengths)
            support_loss = criterion(support_logits, support_labels)

            # Compute gradients normally
            meta_model.zero_grad()
            support_loss.backward()

            # Manual SGD update (detach gradients for FOMAML)
            with torch.no_grad():
                for param in meta_model.parameters():
                    if param.grad is not None:
                        param.data = param.data - inner_lr * param.grad.data
"""

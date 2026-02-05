import torch
import sys
sys.path.insert(0, "/workspace")

from src.models.set_only.set_only_lm import SetOnlyLM

config = {
    "vocab_size": 1000,
    "d_model": 64,
    "num_layers": 2,
    "num_heads": 4,
    "window_size": 32,
    "stride": 32,
    "dropout": 0.0,
    "max_seq_len": 128,
    "backend": "dense_exact",
    "router_topk": 2,
    "feature_mode": "hashed_counts",
    "router_type": "learned",
    "pooling": {"mode": "mean"},
    "geometry": {"enabled": True},
    "sig_gating": {"enabled": False},
}

print("Creating model...")
model = SetOnlyLM(**config)

print("\n=== Testing in eval mode ===")
model.eval()
input_ids = torch.randint(0, 1000, (2, 128))

try:
    with torch.no_grad():
        logits = model(input_ids)
    print(f"Eval logits: shape={logits.shape}, has_nan={logits.isnan().any()}, has_inf={logits.isinf().any()}")
except Exception as e:
    print(f"Eval ERROR: {e}")
    import traceback
    traceback.print_exc()

print("\n=== Testing in train mode ===")
model.train()
input_ids = torch.randint(0, 1000, (2, 128))

try:
    logits = model(input_ids)
    print(f"Train logits: shape={logits.shape}, has_nan={logits.isnan().any()}, has_inf={logits.isinf().any()}")
    
    # Compute loss
    target = torch.randint(0, 1000, (2, 128))
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, 1000),
        target.reshape(-1)
    )
    print(f"CE Loss: {loss.item():.4f}, is_nan={loss.isnan()}")
    
    # Add diversity loss
    from src.models.set_only.losses import set_diversity_loss
    set_embs = model.get_last_set_embeddings()
    if set_embs is not None:
        div_loss = set_diversity_loss(set_embs, target_similarity=0.3)
        print(f"Diversity loss: {div_loss.item():.4f}, is_nan={div_loss.isnan()}")
        total_loss = loss + 0.01 * div_loss
    else:
        total_loss = loss
    
    print(f"Total loss: {total_loss.item():.4f}, is_nan={total_loss.isnan()}")
    
    # Backward
    total_loss.backward()
    
    # Check gradients
    if hasattr(model.router, "temperature"):
        temp_grad = model.router.temperature.grad
        print(f"Router temperature grad: {temp_grad}")
        if temp_grad is not None:
            print(f"  has_nan={temp_grad.isnan().any()}, has_inf={temp_grad.isinf().any()}")
    
    query_grad = model.router.query.weight.grad
    if query_grad is not None:
        print(f"Router query grad norm: {query_grad.norm().item():.6f}")
    
    print("\n✅ Training loop works!")
    
except Exception as e:
    print(f"Train ERROR: {e}")
    import traceback
    traceback.print_exc()

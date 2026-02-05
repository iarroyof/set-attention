import torch
import sys
sys.path.insert(0, "/workspace")

from src.models.set_only.banks import build_window_bank

# Test if all tokens have valid sets
seq_len = 128
bank = build_window_bank(seq_len=seq_len, window_size=32, stride=32, device=torch.device("cpu"))

print(f"Bank shape: {bank.token_to_sets.shape}")
print(f"Number of tokens: {seq_len}")
print(f"Number of sets: {bank.set_indices.shape[0]}")

# Check if any token has no sets
tokens_without_sets = []
for i in range(seq_len):
    if (bank.token_to_sets[i] >= 0).sum() == 0:
        tokens_without_sets.append(i)

if tokens_without_sets:
    print(f"\n❌ FOUND THE BUG: {len(tokens_without_sets)} tokens have NO sets!")
    print(f"Tokens without sets: {tokens_without_sets[:10]}")
else:
    print("\n✅ All tokens belong to at least one set")

# Check distribution
sets_per_token = (bank.token_to_sets >= 0).sum(dim=1)
print(f"\nSets per token: min={sets_per_token.min()}, max={sets_per_token.max()}, mean={sets_per_token.float().mean():.2f}")

# Show a few examples
print(f"\nFirst 10 tokens' sets:")
for i in range(10):
    valid = bank.token_to_sets[i][bank.token_to_sets[i] >= 0]
    print(f"  Token {i}: sets {valid.tolist()}")

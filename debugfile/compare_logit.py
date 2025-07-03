import torch
import os

def main():
    file1 = "llm_outputs_train.pt"
    file2 = "llm_outputs_test.pt"

    sys_path = os.path.dirname(os.path.abspath(__file__))

    # Load tensors
    tensor1 = torch.load(os.path.join(sys_path, file1))
    tensor2 = torch.load(os.path.join(sys_path, file2))


    tensor1 = tensor1.to('cpu').to(torch.float32)
    tensor2 = tensor2.to('cpu').to(torch.float32)

    # Check shape
    if tensor1.shape != tensor2.shape:
        print(f"Shapes differ: {tensor1.shape} vs {tensor2.shape}")
        # return

    # special case for llm inputs and outputs
    tensor1 = tensor1[:,0:tensor2.shape[1],:]
    # tensor2 = tensor2[:,:-50,:]

    # Per-token cosine similarity
    # Shape is [1, N, C] where N is number of tokens, C is feature dimension
    batch_size, seq_len, feature_dim = tensor1.shape  # batch_size=1, seq_len=N, feature_dim=C
    
    # Reshape to [N, C] for easier computation (since batch_size=1)
    tensor1_reshaped = tensor1.view(-1, feature_dim).double()  # [N, C]
    tensor2_reshaped = tensor2.view(-1, feature_dim).double()  # [N, C]
    
    # Compute cosine similarity for each token
    token_cosine_sims = torch.nn.functional.cosine_similarity(
        tensor1_reshaped, tensor2_reshaped, dim=1
    )  # [N] - one similarity score per token
    
    # Define threshold for "different" tokens
    similarity_threshold = 0.95  # tokens with cosine similarity < 0.95 are considered different
    
    # Count different tokens
    different_tokens = (token_cosine_sims < similarity_threshold).sum()
    total_tokens = batch_size * seq_len
    
    print(f"Per-token cosine similarity statistics:")
    print(f"Mean cosine similarity: {token_cosine_sims.mean().item():.6f}")
    print(f"Min cosine similarity: {token_cosine_sims.min().item():.6f}")
    print(f"Max cosine similarity: {token_cosine_sims.max().item():.6f}")
    print(f"Different tokens (< {similarity_threshold}): {different_tokens.item()}")
    print(f"Total tokens: {total_tokens}")
    print(f"Percentage of different tokens: {(different_tokens.item() / total_tokens * 100):.2f}%")


if __name__ == "__main__":
    main()

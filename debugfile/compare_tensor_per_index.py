import torch
import os
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import tqdm

def main():
    file1 = "image_features_train.pt"
    file2 = "image_features_test.pt"

    sys_path = os.path.dirname(os.path.abspath(__file__))

    # Load tensors
    tensor1 = torch.load(os.path.join(sys_path, file1))
    tensor2 = torch.load(os.path.join(sys_path, file2))

    if isinstance(tensor1, dict):
        tensor1 = tensor1['image'][0]
    if isinstance(tensor2, dict):
        tensor2 = tensor2['image'][0]

    tensor1 = tensor1.to('cpu').to(torch.float32).detach().double()
    tensor2 = tensor2.to('cpu').to(torch.float32).detach().double()

    # Check shape
    if tensor1.shape != tensor2.shape:
        print(f"Shapes differ: {tensor1.shape} vs {tensor2.shape}")
        return

    print(f"Tensor shapes: {tensor1.shape}")
    
    # Ensure we have at least 2D tensors (N, C)
    if len(tensor1.shape) < 2:
        print("Error: Tensors must have at least 2 dimensions (N, C)")
        return
    
    # For image features, X,N,C, X is from dynamic s2
    if len(tensor1.shape) == 3:
        original_shape = tensor1.shape
        tensor1 = tensor1.reshape(-1, tensor1.shape[-1])
        tensor2 = tensor2.reshape(-1, tensor2.shape[-1])
        print(f"Reshaped from {original_shape} to {tensor1.shape}")

    # for input images, X,3,448,448
    if len(tensor1.shape) == 4:
        original_shape = tensor1.shape
        tensor1 = tensor1.permute(0,2,3,1).reshape(-1, tensor1.shape[1])
        tensor2 = tensor2.permute(0,2,3,1).reshape(-1, tensor2.shape[1])
        print(f"Reshaped from {original_shape} to {tensor1.shape}")

    N, C = tensor1.shape
    print(f"Computing per-index cosine similarity for {N} samples with {C} dimensions each")
    print("=" * 70)

    # Compute cosine similarity for each index
    per_index_similarities = []
    
    for i in tqdm.tqdm(range(N)):
        # Get the i-th sample from both tensors
        sample1 = tensor1[i].unsqueeze(0)  # Shape: (1, C)
        sample2 = tensor2[i].unsqueeze(0)  # Shape: (1, C)
        
        # Compute cosine similarity for this pair
        cos_sim = F.cosine_similarity(sample1, sample2, dim=1).item()
        per_index_similarities.append(cos_sim)

    per_index_similarities = np.array(per_index_similarities)

    # Statistics
    print("Per-Index Cosine Similarity Statistics:")
    print(f"Mean: {per_index_similarities.mean():.6f}")
    print(f"Std:  {per_index_similarities.std():.6f}")
    print(f"Min:  {per_index_similarities.min():.6f}")
    print(f"Max:  {per_index_similarities.max():.6f}")
    print(f"Median: {np.median(per_index_similarities):.6f}")

    # Distribution analysis
    print("\nDistribution Analysis:")
    very_similar = (per_index_similarities > 0.95).sum()
    similar = ((per_index_similarities > 0.8) & (per_index_similarities <= 0.95)).sum()
    moderate = ((per_index_similarities > 0.5) & (per_index_similarities <= 0.8)).sum()
    dissimilar = (per_index_similarities <= 0.5).sum()
    
    print(f"Very similar (>0.95):     {very_similar:4d} ({very_similar/N*100:.1f}%)")
    print(f"Similar (0.8-0.95]:       {similar:4d} ({similar/N*100:.1f}%)")
    print(f"Moderate (0.5-0.8]:       {moderate:4d} ({moderate/N*100:.1f}%)")
    print(f"Dissimilar (≤0.5):        {dissimilar:4d} ({dissimilar/N*100:.1f}%)")

    # Find most and least similar indices
    print("\nMost and Least Similar Indices:")
    sorted_indices = np.argsort(per_index_similarities)
    
    print("Top 10 MOST similar indices:")
    for i in range(min(10, N)):
        idx = sorted_indices[-(i+1)]
        sim = per_index_similarities[idx]
        print(f"  Index {idx:4d}: {sim:.6f}")
    
    print("\nTop 10 LEAST similar indices:")
    for i in range(min(10, N)):
        idx = sorted_indices[i]
        sim = per_index_similarities[idx]
        print(f"  Index {idx:4d}: {sim:.6f}")

if __name__ == "__main__":
    main() 
#!/usr/bin/env python3
"""
Simple test script to verify the OpenPCS implementation works.
This demonstrates the core functions without requiring actual data.
"""

import numpy as np
from openpcs import fast_logdet, score_loglike, cov_matrix_identity

def test_basic_functions():
    """Test the basic mathematical functions."""
    print("Testing basic functions...")

    # Test fast_logdet
    matrix = np.array([[2.0, 1.0], [1.0, 2.0]])
    log_det = fast_logdet(matrix)
    print(f"Log determinant of [[2,1],[1,2]]: {log_det}")

    # Test score_loglike
    data = np.array([[1.0, 2.0], [3.0, 4.0]])
    cov_matrix = np.eye(2)
    scores = score_loglike(data, cov_matrix)
    print(f"Log-likelihood scores: {scores}")

    # Test cov_matrix_identity
    features = np.random.randn(10, 4)
    cov_matrix = np.cov(features.T)
    features_transformed, cov_identity = cov_matrix_identity(features, cov_matrix)
    print(f"Original cov shape: {cov_matrix.shape}, Identity cov shape: {cov_identity.shape}")
    print("✓ Basic functions working")

def test_pca_model():
    """Test PCA model fitting (mock data)."""
    print("\nTesting PCA model fitting...")

    from openpcs import fit_pca_model

    # Create mock data
    np.random.seed(42)
    feat_np = np.random.randn(100, 64)  # 100 samples, 64 features
    true_np = np.random.randint(0, 3, 100)  # 3 classes
    prds_np = true_np.copy()  # Perfect predictions for simplicity

    # Fit PCA for class 0
    model, cov_matrix = fit_pca_model(feat_np, true_np, prds_np, 0, n_components=16)
    print(f"PCA model fitted for class 0, components: {model.n_components_}")
    print(f"Covariance matrix shape: {cov_matrix.shape}")
    print("✓ PCA model fitting working")

if __name__ == "__main__":
    print("OpenPCS Implementation Test")
    print("=" * 40)

    try:
        test_basic_functions()
        test_pca_model()
        print("\n" + "=" * 40)
        print("✓ All tests passed! Implementation is working.")
        print("\nTo run the full pipeline, you need:")
        print("1. Dataset with .tif images and mask_train_test_int.png")
        print("2. Run: python main.py --operation train --output_path ./output --dataset_path ./data --images img1 img2 --patch_size 25 --hidden_class 2 --network fcn")

    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
def main():
    import numpy as np
    from gmr import GMM
    from time import time

    # Your dataset as a numpy array
    data1 = np.random.randn(100, 4)
    data2 = np.random.randn(100, 4)
    data3 = np.random.randn(100, 4)
    data4 = np.random.randn(100, 4)

    # Create multiple GMM objects
    gmm1 = GMM(n_components=3)
    gmm2 = GMM(n_components=3)
    gmm3 = GMM(n_components=3)
    gmm4 = GMM(n_components=3)

    # Fit the GMMs to the data
    gmm1.from_samples(data1)
    gmm2.from_samples(data2)
    gmm3.from_samples(data3)
    gmm4.from_samples(data4)

    # Use batch-gmr for parallel GMR
    from batch_gmr import BatchGMM

    gmm_list = [gmm1, gmm2, gmm3, gmm4] * 5000

    # With CUDA
    device = "cpu"  # or "cuda:0"
    b_gmm = BatchGMM(gmm_list=gmm_list, device=device)
    b_x = np.random.randn(len(gmm_list), 2)
    start = time()
    b_cgmm = b_gmm.condition([0, 1], b_x)
    b_out = b_cgmm.one_sample_confidence_region(alpha=0.7)
    end = time()
    print("Batch GMR time: ", round(end - start, 4))

    # Use GMR for sequential GMR
    start = time()
    for idx, gmm in enumerate(gmm_list):
        gmr = gmm.condition([0, 1], b_x[idx])
        out = gmr.sample_confidence_region(n_samples=1, alpha=0.7)
    end = time()
    print("Sequential GMR time: ", round(end - start, 4))


if __name__ == "__main__":
    main()

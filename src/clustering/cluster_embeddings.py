import os
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from collections import Counter

import umap
import matplotlib.pyplot as plt

try:
    import hdbscan
    HAS_HDBSCAN = True
except ImportError:
    HAS_HDBSCAN = False
    print("[WARN] hdbscan not installed; HDBSCAN clustering will be skipped.")

EMB_DIR = "data/processed/bert_embeddings"
FIG_DIR = "reports/figures"


def load_incident_embeds(split="test"):
    X = np.load(os.path.join(EMB_DIR, f"{split}_embeddings.npy"))
    y = np.load(os.path.join(EMB_DIR, f"{split}_labels.npy"))
    meta = pd.read_csv(os.path.join(EMB_DIR, f"{split}_metadata.csv"))
    mapping_df = pd.read_csv(os.path.join(EMB_DIR, "label_mapping.csv"))
    id2label = {row["id"]: row["label"] for _, row in mapping_df.iterrows()}
    y_labels = np.array([id2label[i] for i in y])
    return X, y, y_labels, meta


def cluster_with_kmeans(X, y_labels, n_clusters=6):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X)

    print("\n[KMEANS] Cluster sizes:", Counter(clusters))
    for c in range(n_clusters):
        mask = clusters == c
        cls_counts = Counter(y_labels[mask])
        print(f"\n[KMEANS] Cluster {c}: {mask.sum()} samples")
        for cls, cnt in cls_counts.most_common():
            print(f"  {cls}: {cnt}")

    return clusters


def cluster_with_hdbscan(X, y_labels):
    if not HAS_HDBSCAN:
        return None

    clusterer = hdbscan.HDBSCAN(min_cluster_size=10)
    clusters = clusterer.fit_predict(X)

    print("\n[HDBSCAN] Cluster counts:", Counter(clusters))
    for c in sorted(set(clusters)):
        mask = clusters == c
        cls_counts = Counter(y_labels[mask])
        label = f"cluster {c}" if c != -1 else "noise (-1)"
        print(f"\n[HDBSCAN] {label}: {mask.sum()} samples")
        for cls, cnt in cls_counts.most_common():
            print(f"  {cls}: {cnt}")

    return clusters


def plot_umap(X, labels, title, filename):
    os.makedirs(FIG_DIR, exist_ok=True)
    reducer = umap.UMAP(random_state=42)
    X_2d = reducer.fit_transform(X)

    plt.figure(figsize=(8, 6))
    uniq = sorted(set(labels))
    for u in uniq:
        mask = labels == u
        plt.scatter(X_2d[mask, 0], X_2d[mask, 1], label=str(u), alpha=0.7, s=20)
    plt.legend()
    plt.title(title)
    plt.tight_layout()
    out_path = os.path.join(FIG_DIR, filename)
    plt.savefig(out_path)
    plt.close()
    print(f"[OK] Saved plot to {out_path}")


def main():
    X_test, y_test_ids, y_test_labels, meta = load_incident_embeds("test")

    print("[INFO] Incident-level embeddings shape:", X_test.shape)

    # 1) UMAP by true fault_class
    plot_umap(
        X_test,
        np.array(y_test_labels),
        title="UMAP of BERT incident embeddings (colored by fault_class)",
        filename="umap_incidents_by_fault.png",
    )

    # 2) K-means clustering
    k_clusters = cluster_with_kmeans(X_test, y_test_labels, n_clusters=6)
    plot_umap(
        X_test,
        k_clusters,
        title="UMAP of BERT incident embeddings (colored by k-means cluster)",
        filename="umap_incidents_by_kmeans.png",
    )

    # 3) HDBSCAN clustering (optional)
    if HAS_HDBSCAN:
        h_clusters = cluster_with_hdbscan(X_test, y_test_labels)
        plot_umap(
            X_test,
            h_clusters,
            title="UMAP of BERT incident embeddings (colored by HDBSCAN cluster)",
            filename="umap_incidents_by_hdbscan.png",
        )


if __name__ == "__main__":
    main()
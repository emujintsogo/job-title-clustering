"""
cluster_job_descriptions.py

Cluster job descriptions using K-means with TF-IDF embeddings.
Finds optimal cluster count, visualizes results, generates cluster labels.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
import joblib

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Quick and dirty way to remove capitalized words not at the start of the sentence
# E.g. company names
def remove_proper_nouns(text):
    return re.sub(r"\b[A-Z][a-zA-Z]{3,}\b", "", text)

def load_data(path):
    """Load job descriptions CSV."""
    print(f"\n📂 Loading data from {path}...")
    df = pd.read_csv(path)
    df = df.dropna(subset=['description'])
    print(f"   Loaded {len(df):,} job postings")
    return df


def create_embeddings(descriptions, max_features=5000):
    """Create TF-IDF embeddings."""
    print(f"\n🔤 Creating TF-IDF embeddings...")

    custom_stopwords = {
    "experience", "work", "working", "must", "ability",
    "requirements", "applicants", "eligible", "responsibilities",
    "qualification", "skills", "you", "we", "team", 
    "you", "your", "we", "our", "they", "their", "them"
    "company", "organization", "opportunity", "services",
    "candidate", "job", "apply", "role", "position",
    "company","organization","services","team"
    }
    
    stopwords = list(ENGLISH_STOP_WORDS.union(custom_stopwords))

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words=stopwords,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95
    )
    embeddings = vectorizer.fit_transform(descriptions).toarray()
    print(f"   Shape: {embeddings.shape}")
    return embeddings, vectorizer


def find_optimal_k(embeddings, min_k=20, max_k=240, step=20, sample_size=1000):
    """Find optimal clusters using elbow + silhouette."""
    print(f"\n🔍 Testing {min_k}-{max_k} clusters (step={step})...")
    
    # Sample for speed if dataset is large
    if len(embeddings) > sample_size:
        print(f"   Sampling {sample_size:,} points for faster computation...")
        indices = np.random.choice(len(embeddings), sample_size, replace=False)
        sample_embeddings = embeddings[indices]
    else:
        sample_embeddings = embeddings
    
    inertias = []
    silhouettes = []
    k_range = range(min_k, max_k + 1, step)
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(sample_embeddings)
        inertias.append(kmeans.inertia_)
        silhouettes.append(silhouette_score(sample_embeddings, labels))
        print(f"   k={k}: Silhouette={silhouettes[-1]:.4f}")
    
    # Pick k with highest silhouette score
    optimal_k = k_range[np.argmax(silhouettes)]
    print(f"\n   🎯 Optimal k: {optimal_k}")
    
    return optimal_k, k_range, inertias, silhouettes


def plot_metrics(k_range, inertias, silhouettes, optimal_k, output_dir):
    """Plot elbow and silhouette curves."""
    print(f"\n📊 Plotting metrics...")
    os.makedirs(output_dir, exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Elbow
    ax1.plot(list(k_range), inertias, 'bo-', linewidth=2)
    ax1.axvline(optimal_k, color='red', linestyle='--', label=f'k={optimal_k}')
    ax1.set_xlabel('Number of Clusters')
    ax1.set_ylabel('Inertia')
    ax1.set_title('Elbow Method')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Silhouette
    ax2.plot(list(k_range), silhouettes, 'go-', linewidth=2)
    ax2.axvline(optimal_k, color='red', linestyle='--', label=f'k={optimal_k}')
    ax2.set_xlabel('Number of Clusters')
    ax2.set_ylabel('Silhouette Score')
    ax2.set_title('Silhouette Analysis')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cluster_metrics.png'), dpi=300)
    plt.close()


def cluster_data(embeddings, n_clusters):
    """Run K-means clustering."""
    print(f"\n🎯 Clustering with k={n_clusters}...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
    labels = kmeans.fit_predict(embeddings)
    
    print(f"   Silhouette: {silhouette_score(embeddings, labels):.4f}")
    print(f"\n   Cluster sizes:")
    unique, counts = np.unique(labels, return_counts=True)
    for cid, count in zip(unique, counts):
        print(f"      Cluster {cid}: {count:,} ({count/len(labels)*100:.1f}%)")
    
    return kmeans, labels


def generate_labels(df, labels, n_clusters): 
    """Generate cluster labels from keywords.""" 
    print(f"\n🏷️ Generating cluster labels...") 

    cluster_info = {} 
    
    for cid in range(n_clusters): 
        # Get all descriptions in this cluster 
        cluster_docs = df[labels == cid]['description'] 
        text = ' '.join(cluster_docs.tolist()).lower() 

        # Extract keywords 
        text = re.sub(r'[^\w\s]', ' ', text) 
        stop_words = {'the', 'and', 'or', 'for', 'with', 'from', 
                      'this', 'that', 'will', 'are', 'has', 'have',
                    'been', 'can', 'may', 'should', 
                    "experience", "work", "working", "must", "ability",
                    "requirements", "applicants", "eligible", "responsibilities",
                    "qualification", "skills", "you", "we", "team", 
                    "you", "your", "we", "our", "they", "their", "them"
                    "company", "organization", "opportunity", "services",
                    "candidate", "job", "apply", "role", "position",
                    "company","organization","services","team"
                    }
        words = [w for w in text.split() if len(w) > 2 and w not in stop_words] 
        top_words = [w for w, _ in Counter(words).most_common(5)] 
        
        # Create label 
        label = ' '.join(top_words[:3]).title() 
        if len(label) > 50: 
            label = label[:47] + '...' 
    
        cluster_info[cid] = {  
            'cluster_id': cid, 
            'label': label, 
            'size': (labels == cid).sum(), 
            'keywords': ', '.join(top_words) 
        } 
        
        print(f" Cluster {cid}: {label}") 
    return cluster_info


def visualize_clusters(embeddings, labels, cluster_info, output_dir):
    """Create 2D and 3D visualizations."""
    print(f"\n📊 Creating visualizations...")
    os.makedirs(output_dir, exist_ok=True)
    
    n_clusters = len(set(labels))
    colors = plt.cm.tab20(np.linspace(0, 1, n_clusters))
    
    # 2D PCA
    pca_2d = PCA(n_components=2, random_state=42)
    reduced_2d = pca_2d.fit_transform(embeddings)
    
    plt.figure(figsize=(16, 10))
    for cid in range(n_clusters):
        mask = labels == cid
        plt.scatter(reduced_2d[mask, 0], reduced_2d[mask, 1],
                   c=[colors[cid]], label=cluster_info[cid]['label'],
                   alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('Job Clusters (PCA)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'clusters_2d.png'), dpi=300)
    plt.close()
    
    # 3D PCA
    pca_3d = PCA(n_components=3, random_state=42)
    reduced_3d = pca_3d.fit_transform(embeddings)
    
    fig = plt.figure(figsize=(16, 10))
    ax = fig.add_subplot(111, projection='3d')
    for cid in range(n_clusters):
        mask = labels == cid
        ax.scatter(reduced_3d[mask, 0], reduced_3d[mask, 1], reduced_3d[mask, 2],
                  c=[colors[cid]], label=cluster_info[cid]['label'],
                  alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    ax.set_title('Job Clusters (3D PCA)')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'clusters_3d.png'), dpi=300)
    plt.close()
    
    # t-SNE (slower but better separation)
    print(f"   Creating t-SNE plot (may take a few minutes)...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    reduced_tsne = tsne.fit_transform(embeddings)
    
    plt.figure(figsize=(16, 10))
    for cid in range(n_clusters):
        mask = labels == cid
        plt.scatter(reduced_tsne[mask, 0], reduced_tsne[mask, 1],
                   c=[colors[cid]], label=cluster_info[cid]['label'],
                   alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.title('Job Clusters (t-SNE)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'clusters_tsne.png'), dpi=300)
    plt.close()


def save_results(df, labels, cluster_info, output_csv, output_labels):
    """Save clustered data and cluster info."""
    print(f"\n💾 Saving results...")
    
    # Add cluster assignments to dataframe
    df['cluster_id'] = labels
    df['cluster_label'] = [cluster_info[cid]['label'] for cid in labels]
    df.to_csv(output_csv, index=False)
    print(f"   ✅ Clustered jobs: {output_csv}")
    
    # Save cluster info
    info_df = pd.DataFrame(cluster_info.values())
    info_df.to_csv(output_labels, index=False)
    print(f"   ✅ Cluster labels: {output_labels}")


if __name__ == "__main__":
    # File paths
    input_csv = os.path.join(BASE_DIR, "data", "recombined_descriptions.csv")
    output_csv = os.path.join(BASE_DIR, "data", "k-means-clustered_jobs.csv")
    output_labels = os.path.join(BASE_DIR, "data", "k-means-cluster_labels.csv")
    output_dir = os.path.join(BASE_DIR, "data", "k-means-clustering_results")
    
    # Clustering params
    MIN_CLUSTERS = 20
    MAX_CLUSTERS = 240
    MAX_FEATURES = 5000
    STEP = 20
    
    print("=" * 70)
    print("JOB CLUSTERING")
    print("=" * 70)
    
    # Load data
    df = load_data(input_csv)

    # Remove proper nouns from the text (e.g. company names)
    df['description'] = df['description'].apply(remove_proper_nouns)

    # Create embeddings
    embeddings, vectorizer = create_embeddings(df['description'], MAX_FEATURES)
    
    # Find optimal k
    optimal_k, k_range, inertias, silhouettes = find_optimal_k(
        embeddings, MIN_CLUSTERS, MAX_CLUSTERS, step=STEP
    )
    plot_metrics(k_range, inertias, silhouettes, optimal_k, output_dir)
    
    # Cluster
    kmeans, labels = cluster_data(embeddings, optimal_k)
    
    # Generate labels
    cluster_info = generate_labels(df, labels, optimal_k)
    
    # Visualize
    visualize_clusters(embeddings, labels, cluster_info, output_dir)
    
    # Save everything
    save_results(df, labels, cluster_info, output_csv, output_labels)
    
    # Save models
    joblib.dump(kmeans, os.path.join(output_dir, 'kmeans_model.pkl'))
    joblib.dump(vectorizer, os.path.join(output_dir, 'vectorizer.pkl'))
    
    print("\n" + "=" * 70)
    print("✅ COMPLETE!")
    print(f"   Clusters: {optimal_k}")
    print(f"   Jobs: {len(df):,}")
    print("=" * 70)
# app.py
import streamlit as st
import numpy as np
import cv2
import os
import random
from skimage.feature import hog, local_binary_pattern, graycomatrix, graycoprops
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import IncrementalPCA, PCA
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, confusion_matrix, silhouette_score
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis
import warnings
warnings.filterwarnings('ignore')

# ---------- Fix random seeds ----------
random.seed(42)
np.random.seed(42)

# ---------- Dataset path ----------
DATASET_PATH = "./cell_images"

# ---------- FUNCTIONS ----------
def load_images_optimized(path, img_size=(96, 96), max_samples=None): 
    images = []
    labels = []
    file_paths = {"Parasitized": [], "Uninfected": []}
    for folder_name in ["Parasitized", "Uninfected"]:
        folder_path = os.path.join(path, folder_name)
        if os.path.exists(folder_path):
            files = os.listdir(folder_path)
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    file_paths[folder_name].append(os.path.join(folder_path, file))
    if max_samples:
        target_count = max_samples // 2
        parasitized_files = random.sample(file_paths["Parasitized"], min(target_count, len(file_paths["Parasitized"])))
        uninfected_files = random.sample(file_paths["Uninfected"], min(target_count, len(file_paths["Uninfected"])))
        all_files_to_load = parasitized_files + uninfected_files
        labels = [0] * len(parasitized_files) + [1] * len(uninfected_files)
    else:
        st.warning("No max_samples specified!")
        return [], []

    for img_path in all_files_to_load:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = cv2.resize(img, img_size)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        img = clahe.apply(img)
        img = cv2.medianBlur(img, 3)
        images.append(img)
        
    combined = list(zip(images, labels))
    random.shuffle(combined)
    images, labels = zip(*combined)
    return np.array(images), np.array(labels)

def extract_hybrid_features(images, batch_size=500):
    all_features = []
    scaler = StandardScaler()
    for start in range(0, len(images), batch_size):
        end = min(start + batch_size, len(images))
        batch = images[start:end]
        batch_features = []
        for img in batch:
            features_list = []
            hog_feat = hog(img, orientations=9, pixels_per_cell=(8, 8),
                            cells_per_block=(3, 3), block_norm='L2-Hys', feature_vector=True)
            features_list.append(hog_feat)
            lbp_1 = local_binary_pattern(img, P=8, R=1, method='uniform')
            lbp_hist_1, _ = np.histogram(lbp_1.ravel(), bins=10, range=(0,10), density=True)
            lbp_2 = local_binary_pattern(img, P=16, R=2, method='uniform')
            lbp_hist_2, _ = np.histogram(lbp_2.ravel(), bins=18, range=(0,18), density=True)
            features_list.append(lbp_hist_1)
            features_list.append(lbp_hist_2)
            img_glcm = (img / 16).astype(np.uint8)
            glcm = graycomatrix(img_glcm, distances=[1,2], angles=[0,np.pi/4,np.pi/2], levels=16, symmetric=True, normed=True)
            glcm_features = []
            for prop in ['contrast','dissimilarity','energy','ASM']:
                glcm_features.extend(graycoprops(glcm, prop).ravel())
            features_list.append(np.array(glcm_features))
            stat_features = [img.mean(), img.std(), img.max()-img.min(), skew(img.ravel()), kurtosis(img.ravel())]
            features_list.append(np.array(stat_features))
            combined = np.concatenate(features_list)
            batch_features.append(combined)
        all_features.extend(batch_features)
    X = np.array(all_features)
    return scaler.fit_transform(X)

def apply_pca_optimized(X, n_components=50, batch_size=500):
    ipca = IncrementalPCA(n_components=n_components)
    for start in range(0, len(X), batch_size):
        end = min(start+batch_size, len(X))
        ipca.partial_fit(X[start:end])
    X_pca = ipca.transform(X)
    variance = np.sum(ipca.explained_variance_ratio_)
    st.write(f"PCA Variance Retained: {variance*100:.2f}% (Dimensions: {X_pca.shape[1]})")
    return X_pca

def map_clusters_to_labels(y_true, y_cluster):
    mapping = {}
    y_true = y_true.astype(int)
    for cluster in np.unique(y_cluster):
        true_labels = y_true[y_cluster==cluster]
        if len(true_labels) > 0:
            majority = np.bincount(true_labels).argmax()
            mapping[cluster] = majority
    mapped_labels = np.vectorize(mapping.get)(y_cluster)
    return mapped_labels

def plot_clusters(X, labels, title="Cluster Visualization"):
    pca_2d = PCA(n_components=2, random_state=42)
    X_2d = pca_2d.fit_transform(X)
    plt.figure(figsize=(8,6))
    scatter = plt.scatter(X_2d[:,0], X_2d[:,1], c=labels, alpha=0.6)
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.title(title)
    plt.colorbar(scatter, label="Cluster")
    st.pyplot(plt)

# ---------- STREAMLIT UI ----------
st.title("🧬 Malaria Cell Image Clustering")
st.write("Clustering Parasitized vs Uninfected cells using K-Means & PCA.")

max_samples = st.slider("Select number of images to load:", min_value=500, max_value=10000, step=500, value=1000)

if st.button("Run Clustering"):
    with st.spinner("Loading images..."):
        images, labels = load_images_optimized(DATASET_PATH, max_samples=max_samples)
    st.success(f"Loaded {len(images)} images")
    
    with st.spinner("Extracting features..."):
        X = extract_hybrid_features(images)
    st.success(f"Features extracted (shape={X.shape})")
    
    with st.spinner("Applying PCA..."):
        X_pca = apply_pca_optimized(X, n_components=50)
    
    with st.spinner("Running K-Means clustering..."):
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=20)
        y_pred = kmeans.fit_predict(X_pca)
        y_mapped = map_clusters_to_labels(labels, y_pred)
    
    st.write("### 🎯 Clustering Results")
    st.write(f"Accuracy (after mapping): {(labels==y_mapped).mean():.4f}")
    st.write(f"Silhouette Score: {silhouette_score(X_pca, y_pred):.4f}")
    
    st.write("### 🖼️ Cluster Visualization")
    plot_clusters(X_pca, y_pred, title="K-Means Clusters (2D PCA)")

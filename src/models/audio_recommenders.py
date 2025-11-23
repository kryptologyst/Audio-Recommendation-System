"""Audio recommendation models."""

import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
import faiss

logger = logging.getLogger(__name__)


class AudioRecommender(ABC):
    """Abstract base class for audio recommendation models."""

    @abstractmethod
    def fit(self, features: np.ndarray, item_ids: List[str]) -> None:
        """Fit the recommendation model.

        Args:
            features: Audio feature matrix
            item_ids: List of item identifiers
        """
        pass

    @abstractmethod
    def recommend(
        self, query_item: str, top_k: int = 10
    ) -> List[Tuple[str, float]]:
        """Get recommendations for a query item.

        Args:
            query_item: Item to get recommendations for
            top_k: Number of recommendations to return

        Returns:
            List of (item_id, score) tuples
        """
        pass


class CosineSimilarityRecommender(AudioRecommender):
    """Content-based recommender using cosine similarity."""

    def __init__(self) -> None:
        """Initialize cosine similarity recommender."""
        self.features = None
        self.item_ids = None
        self.item_to_idx = None

    def fit(self, features: np.ndarray, item_ids: List[str]) -> None:
        """Fit the cosine similarity model.

        Args:
            features: Audio feature matrix
            item_ids: List of item identifiers
        """
        self.features = features
        self.item_ids = item_ids
        self.item_to_idx = {item_id: idx for idx, item_id in enumerate(item_ids)}
        logger.info(f"Fitted cosine similarity model with {len(item_ids)} items")

    def recommend(
        self, query_item: str, top_k: int = 10
    ) -> List[Tuple[str, float]]:
        """Get recommendations using cosine similarity.

        Args:
            query_item: Item to get recommendations for
            top_k: Number of recommendations to return

        Returns:
            List of (item_id, score) tuples
        """
        if query_item not in self.item_to_idx:
            raise ValueError(f"Item {query_item} not found in training data")

        query_idx = self.item_to_idx[query_item]
        query_features = self.features[query_idx:query_idx + 1]

        similarities = cosine_similarity(query_features, self.features)[0]

        # Get top-k similar items (excluding the query item itself)
        similar_indices = np.argsort(similarities)[::-1][1:top_k + 1]
        recommendations = [
            (self.item_ids[idx], similarities[idx])
            for idx in similar_indices
        ]

        return recommendations


class KNNRecommender(AudioRecommender):
    """K-Nearest Neighbors based audio recommender."""

    def __init__(self, n_neighbors: int = 10, metric: str = "cosine") -> None:
        """Initialize KNN recommender.

        Args:
            n_neighbors: Number of neighbors to consider
            metric: Distance metric to use
        """
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.model = NearestNeighbors(
            n_neighbors=n_neighbors + 1,  # +1 to exclude the query item itself
            metric=metric
        )
        self.item_ids = None
        self.item_to_idx = None

    def fit(self, features: np.ndarray, item_ids: List[str]) -> None:
        """Fit the KNN model.

        Args:
            features: Audio feature matrix
            item_ids: List of item identifiers
        """
        self.model.fit(features)
        self.item_ids = item_ids
        self.item_to_idx = {item_id: idx for idx, item_id in enumerate(item_ids)}
        logger.info(f"Fitted KNN model with {len(item_ids)} items")

    def recommend(
        self, query_item: str, top_k: int = 10
    ) -> List[Tuple[str, float]]:
        """Get recommendations using KNN.

        Args:
            query_item: Item to get recommendations for
            top_k: Number of recommendations to return

        Returns:
            List of (item_id, score) tuples
        """
        if query_item not in self.item_to_idx:
            raise ValueError(f"Item {query_item} not found in training data")

        query_idx = self.item_to_idx[query_item]
        query_features = self.features[query_idx:query_idx + 1]

        distances, indices = self.model.kneighbors(query_features)

        # Convert distances to similarities and exclude the query item itself
        similarities = 1 / (1 + distances[0][1:top_k + 1])
        similar_indices = indices[0][1:top_k + 1]

        recommendations = [
            (self.item_ids[idx], similarities[i])
            for i, idx in enumerate(similar_indices)
        ]

        return recommendations


class FAISSRecommender(AudioRecommender):
    """FAISS-based audio recommender for large-scale similarity search."""

    def __init__(self, index_type: str = "IndexFlatIP") -> None:
        """Initialize FAISS recommender.

        Args:
            index_type: Type of FAISS index to use
        """
        self.index_type = index_type
        self.index = None
        self.item_ids = None
        self.item_to_idx = None

    def fit(self, features: np.ndarray, item_ids: List[str]) -> None:
        """Fit the FAISS index.

        Args:
            features: Audio feature matrix
            item_ids: List of item identifiers
        """
        dimension = features.shape[1]
        
        if self.index_type == "IndexFlatIP":
            self.index = faiss.IndexFlatIP(dimension)
        elif self.index_type == "IndexFlatL2":
            self.index = faiss.IndexFlatL2(dimension)
        else:
            raise ValueError(f"Unknown index type: {self.index_type}")

        # Normalize features for inner product similarity
        if self.index_type == "IndexFlatIP":
            features_normalized = features.copy().astype(np.float32)
            faiss.normalize_L2(features_normalized)
            features = features_normalized
        
        self.index.add(features.astype(np.float32))
        self.features = features  # Store features for querying
        self.item_ids = item_ids
        self.item_to_idx = {item_id: idx for idx, item_id in enumerate(item_ids)}
        logger.info(f"Fitted FAISS {self.index_type} with {len(item_ids)} items")

    def recommend(
        self, query_item: str, top_k: int = 10
    ) -> List[Tuple[str, float]]:
        """Get recommendations using FAISS.

        Args:
            query_item: Item to get recommendations for
            top_k: Number of recommendations to return

        Returns:
            List of (item_id, score) tuples
        """
        if query_item not in self.item_to_idx:
            raise ValueError(f"Item {query_item} not found in training data")

        query_idx = self.item_to_idx[query_item]
        query_features = self.features[query_idx:query_idx + 1].astype(np.float32)

        # Normalize query for inner product similarity
        if self.index_type == "IndexFlatIP":
            query_features_normalized = query_features.copy()
            faiss.normalize_L2(query_features_normalized)
            query_features = query_features_normalized

        scores, indices = self.index.search(query_features, top_k + 1)

        # Exclude the query item itself and convert to recommendations
        recommendations = []
        for i, (idx, score) in enumerate(zip(indices[0], scores[0])):
            if idx != query_idx:  # Skip the query item itself
                recommendations.append((self.item_ids[idx], float(score)))

        return recommendations[:top_k]


class PCAReducer:
    """PCA-based dimensionality reduction for audio features."""

    def __init__(self, n_components: int = 50) -> None:
        """Initialize PCA reducer.

        Args:
            n_components: Number of components to keep
        """
        self.n_components = n_components
        self.pca = PCA(n_components=n_components)

    def fit_transform(self, features: np.ndarray) -> np.ndarray:
        """Fit PCA and transform features.

        Args:
            features: Input feature matrix

        Returns:
            Reduced feature matrix
        """
        return self.pca.fit_transform(features)

    def transform(self, features: np.ndarray) -> np.ndarray:
        """Transform features using fitted PCA.

        Args:
            features: Input feature matrix

        Returns:
            Reduced feature matrix
        """
        return self.pca.transform(features)

    def get_explained_variance_ratio(self) -> np.ndarray:
        """Get explained variance ratio for each component.

        Returns:
            Explained variance ratio array
        """
        return self.pca.explained_variance_ratio_

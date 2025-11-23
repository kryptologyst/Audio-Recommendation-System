"""Evaluation metrics for audio recommendation systems."""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score

logger = logging.getLogger(__name__)


class RecommendationEvaluator:
    """Evaluate audio recommendation models."""

    def __init__(self, k_values: List[int] = [5, 10, 20]) -> None:
        """Initialize evaluator.

        Args:
            k_values: List of k values for top-k metrics
        """
        self.k_values = k_values

    def precision_at_k(
        self, 
        recommendations: List[str], 
        relevant_items: List[str], 
        k: int
    ) -> float:
        """Calculate Precision@K.

        Args:
            recommendations: List of recommended item IDs
            relevant_items: List of relevant item IDs
            k: Number of top recommendations to consider

        Returns:
            Precision@K score
        """
        if k == 0:
            return 0.0
        
        top_k_recs = recommendations[:k]
        relevant_in_top_k = len(set(top_k_recs) & set(relevant_items))
        
        return relevant_in_top_k / k

    def recall_at_k(
        self, 
        recommendations: List[str], 
        relevant_items: List[str], 
        k: int
    ) -> float:
        """Calculate Recall@K.

        Args:
            recommendations: List of recommended item IDs
            relevant_items: List of relevant item IDs
            k: Number of top recommendations to consider

        Returns:
            Recall@K score
        """
        if len(relevant_items) == 0:
            return 0.0
        
        top_k_recs = recommendations[:k]
        relevant_in_top_k = len(set(top_k_recs) & set(relevant_items))
        
        return relevant_in_top_k / len(relevant_items)

    def ndcg_at_k(
        self, 
        recommendations: List[str], 
        relevant_items: List[str], 
        k: int
    ) -> float:
        """Calculate NDCG@K.

        Args:
            recommendations: List of recommended item IDs
            relevant_items: List of relevant item IDs
            k: Number of top recommendations to consider

        Returns:
            NDCG@K score
        """
        if k == 0:
            return 0.0
        
        top_k_recs = recommendations[:k]
        
        # Calculate DCG
        dcg = 0.0
        for i, item in enumerate(top_k_recs):
            if item in relevant_items:
                dcg += 1.0 / np.log2(i + 2)  # +2 because log2(1) = 0
        
        # Calculate IDCG (ideal DCG)
        idcg = 0.0
        for i in range(min(k, len(relevant_items))):
            idcg += 1.0 / np.log2(i + 2)
        
        return dcg / idcg if idcg > 0 else 0.0

    def map_at_k(
        self, 
        recommendations: List[str], 
        relevant_items: List[str], 
        k: int
    ) -> float:
        """Calculate MAP@K.

        Args:
            recommendations: List of recommended item IDs
            relevant_items: List of relevant item IDs
            k: Number of top recommendations to consider

        Returns:
            MAP@K score
        """
        if len(relevant_items) == 0:
            return 0.0
        
        top_k_recs = recommendations[:k]
        precision_sum = 0.0
        relevant_count = 0
        
        for i, item in enumerate(top_k_recs):
            if item in relevant_items:
                relevant_count += 1
                precision_sum += relevant_count / (i + 1)
        
        return precision_sum / len(relevant_items)

    def hit_rate_at_k(
        self, 
        recommendations: List[str], 
        relevant_items: List[str], 
        k: int
    ) -> float:
        """Calculate Hit Rate@K.

        Args:
            recommendations: List of recommended item IDs
            relevant_items: List of relevant item IDs
            k: Number of top recommendations to consider

        Returns:
            Hit Rate@K score
        """
        if len(relevant_items) == 0:
            return 0.0
        
        top_k_recs = recommendations[:k]
        hits = len(set(top_k_recs) & set(relevant_items))
        
        return 1.0 if hits > 0 else 0.0

    def evaluate_user(
        self,
        user_id: str,
        recommendations: List[str],
        relevant_items: List[str]
    ) -> Dict[str, float]:
        """Evaluate recommendations for a single user.

        Args:
            user_id: User identifier
            recommendations: List of recommended item IDs
            relevant_items: List of relevant item IDs

        Returns:
            Dictionary of metric scores
        """
        metrics = {}
        
        for k in self.k_values:
            metrics[f"precision@{k}"] = self.precision_at_k(
                recommendations, relevant_items, k
            )
            metrics[f"recall@{k}"] = self.recall_at_k(
                recommendations, relevant_items, k
            )
            metrics[f"ndcg@{k}"] = self.ndcg_at_k(
                recommendations, relevant_items, k
            )
            metrics[f"map@{k}"] = self.map_at_k(
                recommendations, relevant_items, k
            )
            metrics[f"hit_rate@{k}"] = self.hit_rate_at_k(
                recommendations, relevant_items, k
            )
        
        return metrics

    def evaluate_model(
        self,
        model_predictions: Dict[str, List[str]],
        test_interactions: pd.DataFrame,
        user_col: str = "user_id",
        item_col: str = "item_id"
    ) -> Dict[str, float]:
        """Evaluate model on test set.

        Args:
            model_predictions: Dictionary mapping user_id to recommendations
            test_interactions: Test interactions DataFrame
            user_col: Name of user column
            item_col: Name of item column

        Returns:
            Dictionary of average metric scores
        """
        user_metrics = []
        
        for user_id in test_interactions[user_col].unique():
            user_interactions = test_interactions[
                test_interactions[user_col] == user_id
            ]
            relevant_items = user_interactions[item_col].tolist()
            
            if user_id in model_predictions:
                recommendations = model_predictions[user_id]
                metrics = self.evaluate_user(user_id, recommendations, relevant_items)
                user_metrics.append(metrics)
        
        if not user_metrics:
            logger.warning("No valid predictions found")
            return {}
        
        # Calculate average metrics
        avg_metrics = {}
        for metric in user_metrics[0].keys():
            avg_metrics[metric] = np.mean([m[metric] for m in user_metrics])
        
        return avg_metrics

    def calculate_coverage(
        self,
        model_predictions: Dict[str, List[str]],
        catalog_items: List[str]
    ) -> float:
        """Calculate catalog coverage.

        Args:
            model_predictions: Dictionary mapping user_id to recommendations
            catalog_items: List of all items in catalog

        Returns:
            Coverage score
        """
        recommended_items = set()
        for recommendations in model_predictions.values():
            recommended_items.update(recommendations)
        
        return len(recommended_items) / len(catalog_items)

    def calculate_diversity(
        self,
        model_predictions: Dict[str, List[str]],
        item_features: Optional[np.ndarray] = None,
        item_ids: Optional[List[str]] = None
    ) -> float:
        """Calculate recommendation diversity.

        Args:
            model_predictions: Dictionary mapping user_id to recommendations
            item_features: Feature matrix for items
            item_ids: List of item IDs corresponding to features

        Returns:
            Diversity score (average pairwise distance)
        """
        if item_features is None or item_ids is None:
            # Simple diversity based on unique items
            all_recommendations = []
            for recommendations in model_predictions.values():
                all_recommendations.extend(recommendations)
            
            unique_items = len(set(all_recommendations))
            total_items = len(all_recommendations)
            
            return unique_items / total_items if total_items > 0 else 0.0
        
        # Feature-based diversity
        from sklearn.metrics.pairwise import cosine_similarity
        
        diversity_scores = []
        for recommendations in model_predictions.values():
            if len(recommendations) < 2:
                continue
            
            # Get features for recommended items
            rec_indices = [item_ids.index(item) for item in recommendations if item in item_ids]
            if len(rec_indices) < 2:
                continue
            
            rec_features = item_features[rec_indices]
            similarities = cosine_similarity(rec_features)
            
            # Calculate average pairwise distance (1 - similarity)
            mask = np.triu(np.ones_like(similarities, dtype=bool), k=1)
            pairwise_distances = 1 - similarities[mask]
            diversity_scores.append(np.mean(pairwise_distances))
        
        return np.mean(diversity_scores) if diversity_scores else 0.0

    def calculate_novelty(
        self,
        model_predictions: Dict[str, List[str]],
        item_popularity: Dict[str, float]
    ) -> float:
        """Calculate recommendation novelty.

        Args:
            model_predictions: Dictionary mapping user_id to recommendations
            item_popularity: Dictionary mapping item_id to popularity score

        Returns:
            Novelty score (average negative log popularity)
        """
        novelty_scores = []
        
        for recommendations in model_predictions.values():
            for item in recommendations:
                if item in item_popularity:
                    popularity = item_popularity[item]
                    # Avoid log(0)
                    popularity = max(popularity, 1e-10)
                    novelty_scores.append(-np.log2(popularity))
        
        return np.mean(novelty_scores) if novelty_scores else 0.0

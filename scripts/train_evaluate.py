"""Main training and evaluation script for audio recommendation system."""

import argparse
import logging
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml
from sklearn.preprocessing import StandardScaler

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from audio.feature_extractor import AudioFeatureExtractor
from data.pipeline import AudioDatasetGenerator, DataLoader, DataSplitter
from evaluation.metrics import RecommendationEvaluator
from models.audio_recommenders import (
    CosineSimilarityRecommender,
    FAISSRecommender,
    KNNRecommender,
    PCAReducer,
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def set_random_seeds(seed: int = 42) -> None:
    """Set random seeds for reproducibility.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)


def generate_dataset(
    data_dir: Path,
    n_items: int = 1000,
    n_users: int = 100,
    interactions_per_user: int = 50
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Generate synthetic audio dataset.

    Args:
        data_dir: Directory to save data
        n_items: Number of audio items
        n_users: Number of users
        interactions_per_user: Average interactions per user

    Returns:
        Tuple of (items_df, interactions_df, features)
    """
    logger.info("Generating synthetic audio dataset...")
    
    generator = AudioDatasetGenerator(random_state=42)
    
    # Generate items metadata
    items_df = generator.generate_audio_metadata(n_items=n_items)
    
    # Generate user interactions
    interactions_df = generator.generate_user_interactions(
        items_df, n_users=n_users, interactions_per_user=interactions_per_user
    )
    
    # Generate audio features
    features = generator.generate_audio_features(items_df)
    
    # Save dataset
    data_dir.mkdir(parents=True, exist_ok=True)
    items_df.to_csv(data_dir / "items.csv", index=False)
    interactions_df.to_csv(data_dir / "interactions.csv", index=False)
    np.save(data_dir / "features.npy", features)
    
    logger.info(f"Generated dataset: {len(items_df)} items, {len(interactions_df)} interactions")
    
    return items_df, interactions_df, features


def load_dataset(data_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Load dataset from files.

    Args:
        data_dir: Directory containing data files

    Returns:
        Tuple of (items_df, interactions_df, features)
    """
    logger.info("Loading dataset...")
    
    loader = DataLoader(data_dir)
    items_df = loader.load_items()
    interactions_df = loader.load_interactions()
    features = loader.load_features()
    
    logger.info(f"Loaded dataset: {len(items_df)} items, {len(interactions_df)} interactions")
    
    return items_df, interactions_df, features


def train_models(
    features: np.ndarray,
    item_ids: List[str],
    train_interactions: pd.DataFrame,
    config: Dict
) -> Dict[str, object]:
    """Train multiple recommendation models.

    Args:
        features: Audio feature matrix
        item_ids: List of item identifiers
        train_interactions: Training interactions
        config: Configuration dictionary

    Returns:
        Dictionary of trained models
    """
    logger.info("Training recommendation models...")
    
    models = {}
    
    # Scale features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    # Cosine Similarity Recommender
    if config.get("cosine_similarity", True):
        logger.info("Training Cosine Similarity model...")
        cosine_model = CosineSimilarityRecommender()
        cosine_model.fit(scaled_features, item_ids)
        models["cosine_similarity"] = cosine_model
    
    # KNN Recommender
    if config.get("knn", True):
        logger.info("Training KNN model...")
        knn_model = KNNRecommender(
            n_neighbors=config.get("knn_neighbors", 10),
            metric=config.get("knn_metric", "cosine")
        )
        knn_model.fit(scaled_features, item_ids)
        models["knn"] = knn_model
    
    # FAISS Recommender
    if config.get("faiss", True):
        logger.info("Training FAISS model...")
        faiss_model = FAISSRecommender(
            index_type=config.get("faiss_index_type", "IndexFlatIP")
        )
        faiss_model.fit(scaled_features, item_ids)
        models["faiss"] = faiss_model
    
    # PCA + Cosine Similarity
    if config.get("pca", True):
        logger.info("Training PCA + Cosine Similarity model...")
        pca_reducer = PCAReducer(n_components=config.get("pca_components", 50))
        pca_features = pca_reducer.fit_transform(scaled_features)
        
        pca_cosine_model = CosineSimilarityRecommender()
        pca_cosine_model.fit(pca_features, item_ids)
        models["pca_cosine"] = pca_cosine_model
    
    logger.info(f"Trained {len(models)} models")
    return models


def evaluate_models(
    models: Dict[str, object],
    test_interactions: pd.DataFrame,
    items_df: pd.DataFrame,
    features: np.ndarray,
    config: Dict
) -> Dict[str, Dict[str, float]]:
    """Evaluate all trained models.

    Args:
        models: Dictionary of trained models
        test_interactions: Test interactions
        items_df: Items metadata
        features: Audio feature matrix
        config: Configuration dictionary

    Returns:
        Dictionary of evaluation results
    """
    logger.info("Evaluating models...")
    
    evaluator = RecommendationEvaluator(k_values=config.get("k_values", [5, 10, 20]))
    
    # Get test users
    test_users = test_interactions["user_id"].unique()
    
    results = {}
    
    for model_name, model in models.items():
        logger.info(f"Evaluating {model_name}...")
        
        # Generate recommendations for test users
        predictions = {}
        for user_id in test_users:
            # Get user's training items for recommendation
            user_items = test_interactions[
                test_interactions["user_id"] == user_id
            ]["item_id"].tolist()
            
            if user_items:
                # Use first item as query (in practice, you'd use user's profile)
                query_item = user_items[0]
                try:
                    recommendations = model.recommend(query_item, top_k=20)
                    predictions[user_id] = [item for item, _ in recommendations]
                except Exception as e:
                    logger.warning(f"Error generating recommendations for user {user_id}: {e}")
                    predictions[user_id] = []
        
        # Evaluate model
        metrics = evaluator.evaluate_model(
            predictions, test_interactions
        )
        
        # Calculate additional metrics
        catalog_items = items_df["item_id"].tolist()
        coverage = evaluator.calculate_coverage(predictions, catalog_items)
        
        # Calculate item popularity for novelty
        item_popularity = {}
        for _, item in items_df.iterrows():
            item_popularity[item["item_id"]] = item["popularity"]
        
        novelty = evaluator.calculate_novelty(predictions, item_popularity)
        
        metrics["coverage"] = coverage
        metrics["novelty"] = novelty
        
        results[model_name] = metrics
        
        logger.info(f"{model_name} - Precision@10: {metrics.get('precision@10', 0):.4f}, "
                   f"Recall@10: {metrics.get('recall@10', 0):.4f}, "
                   f"NDCG@10: {metrics.get('ndcg@10', 0):.4f}")
    
    return results


def create_leaderboard(results: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """Create model leaderboard.

    Args:
        results: Evaluation results

    Returns:
        DataFrame with model comparison
    """
    leaderboard_data = []
    
    for model_name, metrics in results.items():
        row = {"model": model_name}
        row.update(metrics)
        leaderboard_data.append(row)
    
    leaderboard = pd.DataFrame(leaderboard_data)
    
    # Sort by NDCG@10 (or another primary metric)
    if "ndcg@10" in leaderboard.columns:
        leaderboard = leaderboard.sort_values("ndcg@10", ascending=False)
    
    return leaderboard


def main():
    """Main training and evaluation pipeline."""
    parser = argparse.ArgumentParser(description="Audio Recommendation System")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                       help="Path to configuration file")
    parser.add_argument("--data_dir", type=str, default="data/processed",
                       help="Path to data directory")
    parser.add_argument("--generate_data", action="store_true",
                       help="Generate synthetic data")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    args = parser.parse_args()
    
    # Set random seeds
    set_random_seeds(args.seed)
    
    # Load configuration
    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        # Default configuration
        config = {
            "dataset": {
                "n_items": 1000,
                "n_users": 100,
                "interactions_per_user": 50
            },
            "models": {
                "cosine_similarity": True,
                "knn": True,
                "faiss": True,
                "pca": True,
                "knn_neighbors": 10,
                "knn_metric": "cosine",
                "faiss_index_type": "IndexFlatIP",
                "pca_components": 50
            },
            "evaluation": {
                "k_values": [5, 10, 20],
                "test_size": 0.2,
                "val_size": 0.1
            }
        }
    
    data_dir = Path(args.data_dir)
    
    # Generate or load dataset
    if args.generate_data or not (data_dir / "items.csv").exists():
        items_df, interactions_df, features = generate_dataset(
            data_dir,
            n_items=config["dataset"]["n_items"],
            n_users=config["dataset"]["n_users"],
            interactions_per_user=config["dataset"]["interactions_per_user"]
        )
    else:
        items_df, interactions_df, features = load_dataset(data_dir)
    
    # Split data
    splitter = DataSplitter(
        test_size=config["evaluation"]["test_size"],
        val_size=config["evaluation"]["val_size"]
    )
    
    train_df, val_df, test_df = splitter.split_interactions(interactions_df)
    
    # Train models
    item_ids = items_df["item_id"].tolist()
    models = train_models(features, item_ids, train_df, config["models"])
    
    # Evaluate models
    results = evaluate_models(models, test_df, items_df, features, config["evaluation"])
    
    # Create leaderboard
    leaderboard = create_leaderboard(results)
    
    # Save results
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    leaderboard.to_csv(results_dir / "leaderboard.csv", index=False)
    
    # Print leaderboard
    print("\n" + "="*80)
    print("MODEL LEADERBOARD")
    print("="*80)
    print(leaderboard.to_string(index=False, float_format="%.4f"))
    print("="*80)
    
    logger.info("Training and evaluation completed successfully!")


if __name__ == "__main__":
    main()

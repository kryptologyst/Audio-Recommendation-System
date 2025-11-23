"""Data pipeline for audio recommendation system."""

import logging
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


class AudioDatasetGenerator:
    """Generate synthetic audio dataset for demonstration purposes."""

    def __init__(self, random_state: int = 42) -> None:
        """Initialize dataset generator.

        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        random.seed(random_state)
        np.random.seed(random_state)

    def generate_audio_metadata(
        self, n_items: int = 1000, n_genres: int = 10
    ) -> pd.DataFrame:
        """Generate synthetic audio metadata.

        Args:
            n_items: Number of audio items to generate
            n_genres: Number of genres to simulate

        Returns:
            DataFrame with audio metadata
        """
        genres = [f"Genre_{i}" for i in range(n_genres)]
        
        data = []
        for i in range(n_items):
            item = {
                "item_id": f"audio_{i:04d}",
                "title": f"Track {i}",
                "artist": f"Artist {i % 50}",
                "genre": random.choice(genres),
                "duration": random.uniform(120, 300),  # 2-5 minutes
                "year": random.randint(1990, 2023),
                "popularity": random.uniform(0, 1),
                "energy": random.uniform(0, 1),
                "valence": random.uniform(0, 1),
                "danceability": random.uniform(0, 1),
            }
            data.append(item)

        return pd.DataFrame(data)

    def generate_user_interactions(
        self, 
        items_df: pd.DataFrame, 
        n_users: int = 100,
        interactions_per_user: int = 50
    ) -> pd.DataFrame:
        """Generate synthetic user interactions.

        Args:
            items_df: DataFrame with item metadata
            n_users: Number of users to simulate
            interactions_per_user: Average interactions per user

        Returns:
            DataFrame with user interactions
        """
        interactions = []
        
        for user_id in range(n_users):
            # Generate user preferences based on genre affinity
            genre_preferences = np.random.dirichlet(np.ones(len(items_df['genre'].unique())))
            genre_weights = dict(zip(items_df['genre'].unique(), genre_preferences))
            
            # Sample items based on genre preferences and popularity
            n_interactions = random.randint(
                interactions_per_user // 2, 
                interactions_per_user * 2
            )
            
            for _ in range(n_interactions):
                # Weight items by genre preference and popularity
                weights = []
                for _, item in items_df.iterrows():
                    genre_weight = genre_weights[item['genre']]
                    popularity_weight = item['popularity']
                    combined_weight = genre_weight * popularity_weight
                    weights.append(combined_weight)
                
                # Sample item based on weights
                item_idx = random.choices(
                    range(len(items_df)), 
                    weights=weights
                )[0]
                
                interaction = {
                    "user_id": f"user_{user_id:04d}",
                    "item_id": items_df.iloc[item_idx]["item_id"],
                    "timestamp": random.randint(1000000000, 1700000000),  # Random timestamp
                    "rating": random.uniform(3, 5),  # Mostly positive ratings
                    "play_count": random.randint(1, 10),
                }
                interactions.append(interaction)

        return pd.DataFrame(interactions)

    def generate_audio_features(
        self, 
        items_df: pd.DataFrame,
        feature_dim: int = 31
    ) -> np.ndarray:
        """Generate synthetic audio features based on metadata.

        Args:
            items_df: DataFrame with item metadata
            feature_dim: Dimension of feature vectors

        Returns:
            Feature matrix
        """
        features = []
        
        for _, item in items_df.iterrows():
            # Create features based on metadata
            feature_vector = np.zeros(feature_dim)
            
            # Genre-based features (first 10 dimensions)
            genre_idx = hash(item['genre']) % 10
            feature_vector[genre_idx] = 1.0
            
            # Energy-based features
            feature_vector[10:15] = np.random.normal(item['energy'], 0.1, 5)
            
            # Valence-based features
            feature_vector[15:20] = np.random.normal(item['valence'], 0.1, 5)
            
            # Danceability-based features
            feature_vector[20:25] = np.random.normal(item['danceability'], 0.1, 5)
            
            # Popularity-based features
            feature_vector[25:30] = np.random.normal(item['popularity'], 0.1, 5)
            
            # Random noise
            feature_vector[30] = np.random.normal(0, 0.1)
            
            features.append(feature_vector)
        
        return np.array(features)


class DataSplitter:
    """Split data into train/validation/test sets."""

    def __init__(self, test_size: float = 0.2, val_size: float = 0.1) -> None:
        """Initialize data splitter.

        Args:
            test_size: Fraction of data for test set
            val_size: Fraction of data for validation set
        """
        self.test_size = test_size
        self.val_size = val_size

    def split_interactions(
        self, 
        interactions_df: pd.DataFrame,
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split interactions into train/val/test sets.

        Args:
            interactions_df: DataFrame with interactions
            random_state: Random seed

        Returns:
            Tuple of (train, val, test) DataFrames
        """
        # First split: train+val vs test
        train_val, test = train_test_split(
            interactions_df,
            test_size=self.test_size,
            random_state=random_state,
            stratify=interactions_df['user_id'] if len(interactions_df['user_id'].unique()) > 1 else None
        )
        
        # Second split: train vs val
        train, val = train_test_split(
            train_val,
            test_size=self.val_size / (1 - self.test_size),
            random_state=random_state,
            stratify=train_val['user_id'] if len(train_val['user_id'].unique()) > 1 else None
        )
        
        logger.info(f"Split interactions: {len(train)} train, {len(val)} val, {len(test)} test")
        
        return train, val, test

    def split_by_time(
        self, 
        interactions_df: pd.DataFrame,
        time_col: str = "timestamp"
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split interactions by time (chronological split).

        Args:
            interactions_df: DataFrame with interactions
            time_col: Name of timestamp column

        Returns:
            Tuple of (train, val, test) DataFrames
        """
        # Sort by timestamp
        sorted_df = interactions_df.sort_values(time_col)
        
        # Split chronologically
        n_total = len(sorted_df)
        n_test = int(n_total * self.test_size)
        n_val = int(n_total * self.val_size)
        n_train = n_total - n_val - n_test
        
        train = sorted_df.iloc[:n_train]
        val = sorted_df.iloc[n_train:n_train + n_val]
        test = sorted_df.iloc[n_train + n_val:]
        
        logger.info(f"Time-based split: {len(train)} train, {len(val)} val, {len(test)} test")
        
        return train, val, test


class DataLoader:
    """Load and preprocess audio recommendation data."""

    def __init__(self, data_dir: Union[str, Path]) -> None:
        """Initialize data loader.

        Args:
            data_dir: Directory containing data files
        """
        self.data_dir = Path(data_dir)

    def load_items(self, filename: str = "items.csv") -> pd.DataFrame:
        """Load items metadata.

        Args:
            filename: Name of items file

        Returns:
            DataFrame with items metadata
        """
        filepath = self.data_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Items file not found: {filepath}")
        
        return pd.read_csv(filepath)

    def load_interactions(self, filename: str = "interactions.csv") -> pd.DataFrame:
        """Load user interactions.

        Args:
            filename: Name of interactions file

        Returns:
            DataFrame with user interactions
        """
        filepath = self.data_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Interactions file not found: {filepath}")
        
        return pd.read_csv(filepath)

    def load_features(self, filename: str = "features.npy") -> np.ndarray:
        """Load pre-computed audio features.

        Args:
            filename: Name of features file

        Returns:
            Feature matrix
        """
        filepath = self.data_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Features file not found: {filepath}")
        
        return np.load(filepath)

    def save_dataset(
        self,
        items_df: pd.DataFrame,
        interactions_df: pd.DataFrame,
        features: np.ndarray,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame
    ) -> None:
        """Save processed dataset.

        Args:
            items_df: Items metadata
            interactions_df: All interactions
            features: Audio features
            train_df: Training interactions
            val_df: Validation interactions
            test_df: Test interactions
        """
        # Save items
        items_df.to_csv(self.data_dir / "items.csv", index=False)
        
        # Save interactions
        interactions_df.to_csv(self.data_dir / "interactions.csv", index=False)
        
        # Save features
        np.save(self.data_dir / "features.npy", features)
        
        # Save splits
        train_df.to_csv(self.data_dir / "train.csv", index=False)
        val_df.to_csv(self.data_dir / "val.csv", index=False)
        test_df.to_csv(self.data_dir / "test.csv", index=False)
        
        logger.info(f"Saved dataset to {self.data_dir}")

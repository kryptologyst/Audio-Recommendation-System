"""Unit tests for audio recommendation system."""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

from src.audio.feature_extractor import AudioFeatureExtractor
from src.data.pipeline import AudioDatasetGenerator, DataSplitter
from src.evaluation.metrics import RecommendationEvaluator
from src.models.audio_recommenders import (
    CosineSimilarityRecommender,
    KNNRecommender,
    FAISSRecommender,
    PCAReducer,
)


class TestAudioFeatureExtractor:
    """Test audio feature extractor."""

    def test_init(self):
        """Test feature extractor initialization."""
        extractor = AudioFeatureExtractor()
        assert extractor.sample_rate == 22050
        assert extractor.n_mfcc == 13
        assert extractor.n_mels == 128

    def test_extract_mfcc_features(self):
        """Test MFCC feature extraction."""
        extractor = AudioFeatureExtractor()
        
        # Mock librosa.load to return dummy audio data
        with patch('librosa.load') as mock_load:
            mock_load.return_value = (np.random.randn(1000), 22050)
            
            with patch('librosa.feature.mfcc') as mock_mfcc:
                mock_mfcc.return_value = np.random.randn(13, 100)
                
                features = extractor.extract_mfcc_features("dummy_path.wav")
                
                assert len(features) == 13
                assert isinstance(features, np.ndarray)

    def test_extract_all_features(self):
        """Test comprehensive feature extraction."""
        extractor = AudioFeatureExtractor()
        
        with patch('librosa.load') as mock_load:
            mock_load.return_value = (np.random.randn(1000), 22050)
            
            with patch('librosa.feature.mfcc') as mock_mfcc:
                mock_mfcc.return_value = np.random.randn(13, 100)
                
                with patch('librosa.feature.spectral_centroid') as mock_centroid:
                    mock_centroid.return_value = np.random.randn(1, 100)
                    
                    with patch('librosa.feature.spectral_rolloff') as mock_rolloff:
                        mock_rolloff.return_value = np.random.randn(1, 100)
                        
                        with patch('librosa.feature.zero_crossing_rate') as mock_zcr:
                            mock_zcr.return_value = np.random.randn(1, 100)
                            
                            with patch('librosa.feature.chroma_stft') as mock_chroma:
                                mock_chroma.return_value = np.random.randn(12, 100)
                                
                                with patch('librosa.beat.beat_track') as mock_beat:
                                    mock_beat.return_value = (120, np.array([0, 100, 200]))
                                    
                                    features = extractor.extract_all_features("dummy_path.wav")
                                    
                                    assert len(features) > 0
                                    assert isinstance(features, np.ndarray)


class TestAudioDatasetGenerator:
    """Test dataset generator."""

    def test_init(self):
        """Test generator initialization."""
        generator = AudioDatasetGenerator(random_state=42)
        assert generator.random_state == 42

    def test_generate_audio_metadata(self):
        """Test audio metadata generation."""
        generator = AudioDatasetGenerator(random_state=42)
        items_df = generator.generate_audio_metadata(n_items=10)
        
        assert len(items_df) == 10
        assert "item_id" in items_df.columns
        assert "title" in items_df.columns
        assert "genre" in items_df.columns
        assert "duration" in items_df.columns

    def test_generate_user_interactions(self):
        """Test user interactions generation."""
        generator = AudioDatasetGenerator(random_state=42)
        items_df = generator.generate_audio_metadata(n_items=10)
        interactions_df = generator.generate_user_interactions(
            items_df, n_users=5, interactions_per_user=3
        )
        
        assert len(interactions_df) > 0
        assert "user_id" in interactions_df.columns
        assert "item_id" in interactions_df.columns
        assert "timestamp" in interactions_df.columns

    def test_generate_audio_features(self):
        """Test audio features generation."""
        generator = AudioDatasetGenerator(random_state=42)
        items_df = generator.generate_audio_metadata(n_items=5)
        features = generator.generate_audio_features(items_df)
        
        assert features.shape[0] == 5
        assert features.shape[1] == 31  # Default feature dimension


class TestDataSplitter:
    """Test data splitter."""

    def test_init(self):
        """Test splitter initialization."""
        splitter = DataSplitter(test_size=0.2, val_size=0.1)
        assert splitter.test_size == 0.2
        assert splitter.val_size == 0.1

    def test_split_interactions(self):
        """Test interaction splitting."""
        splitter = DataSplitter(test_size=0.2, val_size=0.1)
        
        # Create dummy interactions
        interactions_df = pd.DataFrame({
            "user_id": ["user1", "user1", "user2", "user2", "user3", "user3"],
            "item_id": ["item1", "item2", "item1", "item3", "item2", "item3"],
            "timestamp": [1, 2, 3, 4, 5, 6],
            "rating": [4, 5, 3, 4, 5, 4]
        })
        
        train, val, test = splitter.split_interactions(interactions_df)
        
        assert len(train) + len(val) + len(test) == len(interactions_df)
        assert len(train) > 0
        assert len(val) > 0
        assert len(test) > 0


class TestRecommendationEvaluator:
    """Test recommendation evaluator."""

    def test_init(self):
        """Test evaluator initialization."""
        evaluator = RecommendationEvaluator(k_values=[5, 10])
        assert evaluator.k_values == [5, 10]

    def test_precision_at_k(self):
        """Test precision@k calculation."""
        evaluator = RecommendationEvaluator()
        
        recommendations = ["item1", "item2", "item3", "item4", "item5"]
        relevant_items = ["item1", "item3", "item6"]
        
        precision = evaluator.precision_at_k(recommendations, relevant_items, k=5)
        assert precision == 0.4  # 2 relevant items out of 5 recommendations

    def test_recall_at_k(self):
        """Test recall@k calculation."""
        evaluator = RecommendationEvaluator()
        
        recommendations = ["item1", "item2", "item3", "item4", "item5"]
        relevant_items = ["item1", "item3", "item6"]
        
        recall = evaluator.recall_at_k(recommendations, relevant_items, k=5)
        assert recall == 2/3  # 2 relevant items found out of 3 total relevant

    def test_ndcg_at_k(self):
        """Test NDCG@k calculation."""
        evaluator = RecommendationEvaluator()
        
        recommendations = ["item1", "item2", "item3", "item4", "item5"]
        relevant_items = ["item1", "item3", "item6"]
        
        ndcg = evaluator.ndcg_at_k(recommendations, relevant_items, k=5)
        assert 0 <= ndcg <= 1

    def test_hit_rate_at_k(self):
        """Test hit rate@k calculation."""
        evaluator = RecommendationEvaluator()
        
        recommendations = ["item1", "item2", "item3", "item4", "item5"]
        relevant_items = ["item1", "item3", "item6"]
        
        hit_rate = evaluator.hit_rate_at_k(recommendations, relevant_items, k=5)
        assert hit_rate == 1.0  # At least one relevant item found


class TestCosineSimilarityRecommender:
    """Test cosine similarity recommender."""

    def test_init(self):
        """Test recommender initialization."""
        recommender = CosineSimilarityRecommender()
        assert recommender.features is None
        assert recommender.item_ids is None

    def test_fit(self):
        """Test model fitting."""
        recommender = CosineSimilarityRecommender()
        
        features = np.random.randn(5, 10)
        item_ids = ["item1", "item2", "item3", "item4", "item5"]
        
        recommender.fit(features, item_ids)
        
        assert recommender.features is not None
        assert recommender.item_ids == item_ids
        assert len(recommender.item_to_idx) == 5

    def test_recommend(self):
        """Test recommendation generation."""
        recommender = CosineSimilarityRecommender()
        
        features = np.random.randn(5, 10)
        item_ids = ["item1", "item2", "item3", "item4", "item5"]
        
        recommender.fit(features, item_ids)
        
        recommendations = recommender.recommend("item1", top_k=3)
        
        assert len(recommendations) == 3
        assert all(isinstance(score, float) for _, score in recommendations)
        assert all(item in item_ids for item, _ in recommendations)


class TestKNNRecommender:
    """Test KNN recommender."""

    def test_init(self):
        """Test recommender initialization."""
        recommender = KNNRecommender(n_neighbors=5)
        assert recommender.n_neighbors == 5
        assert recommender.metric == "cosine"

    def test_fit_and_recommend(self):
        """Test model fitting and recommendation."""
        recommender = KNNRecommender(n_neighbors=3)
        
        features = np.random.randn(5, 10)
        item_ids = ["item1", "item2", "item3", "item4", "item5"]
        
        recommender.fit(features, item_ids)
        
        recommendations = recommender.recommend("item1", top_k=3)
        
        assert len(recommendations) == 3
        assert all(isinstance(score, float) for _, score in recommendations)


class TestPCAReducer:
    """Test PCA reducer."""

    def test_init(self):
        """Test reducer initialization."""
        reducer = PCAReducer(n_components=5)
        assert reducer.n_components == 5

    def test_fit_transform(self):
        """Test PCA fit and transform."""
        reducer = PCAReducer(n_components=3)
        
        features = np.random.randn(10, 20)
        reduced_features = reducer.fit_transform(features)
        
        assert reduced_features.shape == (10, 3)
        assert isinstance(reducer.get_explained_variance_ratio(), np.ndarray)


if __name__ == "__main__":
    pytest.main([__file__])

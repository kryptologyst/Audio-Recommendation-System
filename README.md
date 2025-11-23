# Audio Recommendation System

A content-based audio recommendation system that uses advanced audio feature extraction techniques to recommend similar music tracks, podcasts, and audiobooks based on acoustic properties.

## Features

- **Advanced Audio Processing**: MFCC coefficients, spectral features, rhythm patterns, and tempo analysis
- **Multiple Recommendation Models**: Cosine similarity, K-Nearest Neighbors, FAISS-based similarity search, and PCA-reduced approaches
- **Comprehensive Evaluation**: Precision@K, Recall@K, NDCG@K, MAP@K, Hit Rate, Coverage, Novelty, and Diversity metrics
- **Interactive Demo**: Streamlit-based user interface for exploring recommendations
- **Production-Ready**: Clean code structure, type hints, comprehensive testing, and proper documentation

## Quick Start

### Installation

1. Clone the repository:
```bash
git clone https://github.com/kryptologyst/Audio-Recommendation-System.git
cd Audio-Recommendation-System
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

Or using pip with the project configuration:
```bash
pip install -e .
```

### Generate Dataset and Train Models

```bash
python scripts/train_evaluate.py --generate_data --config configs/default.yaml
```

### Run Interactive Demo

```bash
streamlit run scripts/demo.py
```

## Project Structure

```
audio-recommendation-system/
├── src/                          # Source code modules
│   ├── audio/                    # Audio processing
│   │   └── feature_extractor.py  # Audio feature extraction
│   ├── data/                     # Data pipeline
│   │   └── pipeline.py          # Dataset generation and loading
│   ├── models/                   # Recommendation models
│   │   └── audio_recommenders.py # Model implementations
│   └── evaluation/               # Evaluation metrics
│       └── metrics.py            # Recommendation metrics
├── scripts/                      # Executable scripts
│   ├── train_evaluate.py         # Training and evaluation
│   └── demo.py                   # Streamlit demo
├── configs/                      # Configuration files
│   └── default.yaml             # Default configuration
├── data/                         # Data directory
│   ├── raw/                      # Raw data files
│   └── processed/                # Processed data files
├── tests/                        # Unit tests
│   └── test_audio_recommendation.py
├── requirements.txt              # Python dependencies
├── pyproject.toml               # Project configuration
└── README.md                    # This file
```

## Dataset Schema

### Items Dataset (`items.csv`)
- `item_id`: Unique identifier for audio item
- `title`: Title of the audio track
- `artist`: Artist name
- `genre`: Musical genre
- `duration`: Duration in seconds
- `year`: Release year
- `popularity`: Popularity score (0-1)
- `energy`: Energy level (0-1)
- `valence`: Valence/mood (0-1)
- `danceability`: Danceability score (0-1)

### Interactions Dataset (`interactions.csv`)
- `user_id`: Unique identifier for user
- `item_id`: Unique identifier for audio item
- `timestamp`: Interaction timestamp
- `rating`: User rating (1-5)
- `play_count`: Number of times played

### Features Dataset (`features.npy`)
- NumPy array containing extracted audio features for each item
- Dimensions: (n_items, n_features)
- Features include MFCC coefficients, spectral features, and rhythm patterns

## Models Implemented

### 1. Cosine Similarity Recommender
- Direct cosine similarity between audio feature vectors
- Fast and interpretable
- Good baseline for content-based recommendations

### 2. K-Nearest Neighbors (KNN)
- KNN-based similarity search using various distance metrics
- Configurable number of neighbors
- Robust to feature scaling

### 3. FAISS Recommender
- Fast similarity search using Facebook's FAISS library
- Supports both inner product and L2 distance
- Scalable to large datasets

### 4. PCA + Cosine Similarity
- Dimensionality reduction using PCA
- Reduces computational complexity
- Maintains most important feature information

## Evaluation Metrics

- **Precision@K**: Fraction of recommended items that are relevant
- **Recall@K**: Fraction of relevant items that are recommended
- **NDCG@K**: Normalized Discounted Cumulative Gain
- **MAP@K**: Mean Average Precision
- **Hit Rate@K**: Fraction of users with at least one relevant recommendation
- **Coverage**: Fraction of catalog items that can be recommended
- **Novelty**: Average negative log popularity of recommended items
- **Diversity**: Average pairwise distance between recommended items

## Configuration

The system can be configured using YAML files in the `configs/` directory:

```yaml
dataset:
  n_items: 1000
  n_users: 100
  interactions_per_user: 50

models:
  cosine_similarity: true
  knn: true
  faiss: true
  pca: true
  knn_neighbors: 10
  knn_metric: "cosine"
  faiss_index_type: "IndexFlatIP"
  pca_components: 50

evaluation:
  k_values: [5, 10, 20]
  test_size: 0.2
  val_size: 0.1
```

## Usage Examples

### Training Models

```python
from src.data.pipeline import AudioDatasetGenerator
from src.models.audio_recommenders import CosineSimilarityRecommender

# Generate dataset
generator = AudioDatasetGenerator(random_state=42)
items_df = generator.generate_audio_metadata(n_items=1000)
interactions_df = generator.generate_user_interactions(items_df)
features = generator.generate_audio_features(items_df)

# Train model
model = CosineSimilarityRecommender()
model.fit(features, items_df["item_id"].tolist())

# Get recommendations
recommendations = model.recommend("audio_0001", top_k=10)
```

### Evaluating Models

```python
from src.evaluation.metrics import RecommendationEvaluator

evaluator = RecommendationEvaluator(k_values=[5, 10, 20])
metrics = evaluator.evaluate_model(predictions, test_interactions)
print(f"Precision@10: {metrics['precision@10']:.4f}")
```

## Interactive Demo

The Streamlit demo provides:

- **Recommendation Interface**: Select an audio track and get similar recommendations
- **Item Search**: Browse and filter audio items by genre
- **Analytics Dashboard**: Visualize audio features and model performance
- **Model Comparison**: Compare different recommendation approaches

To run the demo:
```bash
streamlit run scripts/demo.py
```

## Testing

Run the test suite:
```bash
pytest tests/
```

Run with coverage:
```bash
pytest tests/ --cov=src
```

## Development

### Code Quality

The project uses:
- **Black** for code formatting
- **Ruff** for linting
- **Type hints** for better code documentation
- **Docstrings** following Google style

### Pre-commit Hooks

Install pre-commit hooks:
```bash
pre-commit install
```

### Adding New Models

To add a new recommendation model:

1. Inherit from `AudioRecommender` base class
2. Implement `fit()` and `recommend()` methods
3. Add model configuration to `configs/default.yaml`
4. Update the training script to include your model
5. Add tests for your model

### Adding New Features

To add new audio features:

1. Extend `AudioFeatureExtractor` class
2. Add feature extraction method
3. Update `extract_all_features()` method
4. Update tests

## Performance Considerations

- **Feature Extraction**: Audio feature extraction can be computationally expensive. Consider caching features for large datasets.
- **Model Selection**: FAISS is recommended for large-scale similarity search, while cosine similarity works well for smaller datasets.
- **Memory Usage**: PCA can help reduce memory usage for large feature matrices.

## Future Enhancements

- **Deep Learning Models**: Integration with neural collaborative filtering
- **Real-time Recommendations**: API endpoints for real-time recommendation serving
- **Multi-modal Features**: Integration of lyrics, artist information, and user preferences
- **Cold Start Handling**: Better handling of new users and items
- **A/B Testing Framework**: Built-in experimentation capabilities

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- **librosa**: Audio and music signal processing
- **scikit-learn**: Machine learning algorithms
- **FAISS**: Fast similarity search
- **Streamlit**: Interactive web applications
- **Plotly**: Interactive visualizations
# Audio-Recommendation-System

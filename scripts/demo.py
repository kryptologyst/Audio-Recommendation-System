"""Streamlit demo for audio recommendation system."""

import logging
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from audio.feature_extractor import AudioFeatureExtractor
from data.pipeline import AudioDatasetGenerator, DataLoader
from evaluation.metrics import RecommendationEvaluator
from models.audio_recommenders import (
    CosineSimilarityRecommender,
    FAISSRecommender,
    KNNRecommender,
    PCAReducer,
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Audio Recommendation System",
    page_icon="🎵",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .recommendation-item {
        background-color: #ffffff;
        padding: 0.5rem;
        border-radius: 0.25rem;
        margin: 0.25rem 0;
        border-left: 4px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_dataset(data_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Load dataset with caching."""
    try:
        loader = DataLoader(data_dir)
        items_df = loader.load_items()
        interactions_df = loader.load_interactions()
        features = loader.load_features()
        return items_df, interactions_df, features
    except FileNotFoundError:
        st.error("Dataset not found. Please run the training script first.")
        return None, None, None


@st.cache_data
def generate_sample_dataset() -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Generate sample dataset for demo."""
    generator = AudioDatasetGenerator(random_state=42)
    
    items_df = generator.generate_audio_metadata(n_items=100)
    interactions_df = generator.generate_user_interactions(
        items_df, n_users=20, interactions_per_user=10
    )
    features = generator.generate_audio_features(items_df)
    
    return items_df, interactions_df, features


@st.cache_resource
def load_models(features: np.ndarray, item_ids: List[str]) -> Dict[str, object]:
    """Load trained models with caching."""
    from sklearn.preprocessing import StandardScaler
    
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    models = {}
    
    # Cosine Similarity
    cosine_model = CosineSimilarityRecommender()
    cosine_model.fit(scaled_features, item_ids)
    models["cosine_similarity"] = cosine_model
    
    # KNN
    knn_model = KNNRecommender(n_neighbors=10, metric="cosine")
    knn_model.fit(scaled_features, item_ids)
    models["knn"] = knn_model
    
    # FAISS
    faiss_model = FAISSRecommender(index_type="IndexFlatIP")
    faiss_model.fit(scaled_features, item_ids)
    models["faiss"] = faiss_model
    
    # PCA + Cosine
    pca_reducer = PCAReducer(n_components=20)
    pca_features = pca_reducer.fit_transform(scaled_features)
    pca_cosine_model = CosineSimilarityRecommender()
    pca_cosine_model.fit(pca_features, item_ids)
    models["pca_cosine"] = pca_cosine_model
    
    return models


def display_item_info(item_id: str, items_df: pd.DataFrame) -> None:
    """Display detailed information about an item."""
    item = items_df[items_df["item_id"] == item_id].iloc[0]
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Genre", item["genre"])
        st.metric("Year", int(item["year"]))
    
    with col2:
        st.metric("Energy", f"{item['energy']:.2f}")
        st.metric("Valence", f"{item['valence']:.2f}")
    
    with col3:
        st.metric("Danceability", f"{item['danceability']:.2f}")
        st.metric("Popularity", f"{item['popularity']:.2f}")


def display_recommendations(
    recommendations: List[Tuple[str, float]], 
    items_df: pd.DataFrame,
    model_name: str
) -> None:
    """Display recommendations in a nice format."""
    st.subheader(f"Recommendations from {model_name.replace('_', ' ').title()}")
    
    for i, (item_id, score) in enumerate(recommendations[:10], 1):
        item = items_df[items_df["item_id"] == item_id].iloc[0]
        
        with st.expander(f"{i}. {item['title']} by {item['artist']} (Score: {score:.3f})"):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.write(f"**Genre:** {item['genre']}")
                st.write(f"**Year:** {int(item['year'])}")
                st.write(f"**Duration:** {item['duration']:.0f} seconds")
            
            with col2:
                st.metric("Energy", f"{item['energy']:.2f}")
                st.metric("Valence", f"{item['valence']:.2f}")
                st.metric("Danceability", f"{item['danceability']:.2f}")


def create_feature_visualization(features: np.ndarray, items_df: pd.DataFrame) -> None:
    """Create visualization of audio features."""
    st.subheader("Audio Feature Visualization")
    
    # PCA for visualization
    from sklearn.decomposition import PCA
    
    pca = PCA(n_components=2)
    features_2d = pca.fit_transform(features)
    
    # Create scatter plot
    fig = px.scatter(
        x=features_2d[:, 0],
        y=features_2d[:, 1],
        color=items_df["genre"],
        hover_data={
            "title": items_df["title"],
            "artist": items_df["artist"],
            "genre": items_df["genre"]
        },
        title="Audio Items in Feature Space (PCA)",
        labels={"x": f"PC1 ({pca.explained_variance_ratio_[0]:.1%})", 
                "y": f"PC2 ({pca.explained_variance_ratio_[1]:.1%})"}
    )
    
    st.plotly_chart(fig, use_container_width=True)


def create_metrics_comparison(models: Dict[str, object], test_interactions: pd.DataFrame) -> None:
    """Create metrics comparison visualization."""
    st.subheader("Model Performance Comparison")
    
    evaluator = RecommendationEvaluator(k_values=[5, 10, 20])
    
    # Generate sample predictions for demo
    test_users = test_interactions["user_id"].unique()[:10]  # Limit for demo
    
    model_metrics = {}
    
    for model_name, model in models.items():
        predictions = {}
        for user_id in test_users:
            user_items = test_interactions[
                test_interactions["user_id"] == user_id
            ]["item_id"].tolist()
            
            if user_items:
                query_item = user_items[0]
                try:
                    recommendations = model.recommend(query_item, top_k=20)
                    predictions[user_id] = [item for item, _ in recommendations]
                except:
                    predictions[user_id] = []
        
        metrics = evaluator.evaluate_model(predictions, test_interactions)
        model_metrics[model_name] = metrics
    
    # Create comparison chart
    metrics_to_plot = ["precision@10", "recall@10", "ndcg@10"]
    
    fig = go.Figure()
    
    for metric in metrics_to_plot:
        values = [model_metrics[model].get(metric, 0) for model in models.keys()]
        fig.add_trace(go.Bar(
            name=metric,
            x=list(models.keys()),
            y=values,
            text=[f"{v:.3f}" for v in values],
            textposition="auto"
        ))
    
    fig.update_layout(
        title="Model Performance Comparison",
        xaxis_title="Models",
        yaxis_title="Score",
        barmode="group"
    )
    
    st.plotly_chart(fig, use_container_width=True)


def main():
    """Main Streamlit application."""
    st.markdown('<h1 class="main-header">🎵 Audio Recommendation System</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    This demo showcases a content-based audio recommendation system that uses audio features 
    to recommend similar music tracks. The system extracts features like MFCC coefficients, 
    spectral characteristics, and rhythm patterns to find similar audio content.
    """)
    
    # Sidebar
    st.sidebar.header("Configuration")
    
    # Check if dataset exists
    data_dir = Path("data/processed")
    if (data_dir / "items.csv").exists():
        items_df, interactions_df, features = load_dataset(data_dir)
        st.sidebar.success("✅ Dataset loaded successfully")
    else:
        st.sidebar.warning("⚠️ No dataset found, generating sample data...")
        items_df, interactions_df, features = generate_sample_dataset()
        st.sidebar.info("📊 Sample dataset generated")
    
    if items_df is None:
        st.error("Failed to load dataset. Please check the data directory.")
        return
    
    # Load models
    item_ids = items_df["item_id"].tolist()
    models = load_models(features, item_ids)
    
    # Main interface
    tab1, tab2, tab3, tab4 = st.tabs(["🎯 Recommendations", "🔍 Item Search", "📊 Analytics", "ℹ️ About"])
    
    with tab1:
        st.header("Get Audio Recommendations")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Select Query Item")
            
            # Item selection
            selected_item = st.selectbox(
                "Choose an audio track:",
                options=items_df["item_id"].tolist(),
                format_func=lambda x: f"{items_df[items_df['item_id']==x].iloc[0]['title']} - {items_df[items_df['item_id']==x].iloc[0]['artist']}"
            )
            
            # Display item info
            display_item_info(selected_item, items_df)
            
            # Model selection
            selected_model = st.selectbox(
                "Choose recommendation model:",
                options=list(models.keys()),
                format_func=lambda x: x.replace("_", " ").title()
            )
            
            # Number of recommendations
            top_k = st.slider("Number of recommendations:", 5, 20, 10)
        
        with col2:
            st.subheader("Recommendations")
            
            if st.button("Get Recommendations", type="primary"):
                try:
                    model = models[selected_model]
                    recommendations = model.recommend(selected_item, top_k=top_k)
                    
                    display_recommendations(recommendations, items_df, selected_model)
                    
                except Exception as e:
                    st.error(f"Error generating recommendations: {e}")
    
    with tab2:
        st.header("Search Similar Items")
        
        # Search by genre
        genres = items_df["genre"].unique()
        selected_genre = st.selectbox("Filter by genre:", ["All"] + list(genres))
        
        if selected_genre != "All":
            filtered_items = items_df[items_df["genre"] == selected_genre]
        else:
            filtered_items = items_df
        
        # Display items
        st.subheader("Available Items")
        
        for _, item in filtered_items.iterrows():
            with st.expander(f"{item['title']} - {item['artist']}"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.write(f"**Genre:** {item['genre']}")
                    st.write(f"**Year:** {int(item['year'])}")
                
                with col2:
                    st.write(f"**Energy:** {item['energy']:.2f}")
                    st.write(f"**Valence:** {item['valence']:.2f}")
                
                with col3:
                    st.write(f"**Danceability:** {item['danceability']:.2f}")
                    st.write(f"**Popularity:** {item['popularity']:.2f}")
    
    with tab3:
        st.header("Analytics Dashboard")
        
        # Feature visualization
        create_feature_visualization(features, items_df)
        
        # Metrics comparison
        create_metrics_comparison(models, interactions_df)
        
        # Dataset statistics
        st.subheader("Dataset Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Items", len(items_df))
        
        with col2:
            st.metric("Total Users", len(interactions_df["user_id"].unique()))
        
        with col3:
            st.metric("Total Interactions", len(interactions_df))
        
        with col4:
            st.metric("Genres", len(items_df["genre"].unique()))
        
        # Genre distribution
        genre_counts = items_df["genre"].value_counts()
        fig = px.pie(
            values=genre_counts.values,
            names=genre_counts.index,
            title="Genre Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.header("About This System")
        
        st.markdown("""
        ## Audio Recommendation System
        
        This is a content-based recommendation system specifically designed for audio content 
        (music, podcasts, audiobooks). It uses advanced audio feature extraction techniques 
        to find similar content based on acoustic properties.
        
        ### Features
        
        - **Audio Feature Extraction**: MFCC coefficients, spectral features, rhythm patterns
        - **Multiple Models**: Cosine similarity, KNN, FAISS, PCA-based approaches
        - **Comprehensive Evaluation**: Precision@K, Recall@K, NDCG@K, MAP@K, Hit Rate
        - **Interactive Demo**: Streamlit-based user interface
        
        ### Technical Details
        
        - **Audio Processing**: librosa for feature extraction
        - **Machine Learning**: scikit-learn, FAISS for similarity search
        - **Evaluation**: Custom metrics implementation
        - **Visualization**: Plotly for interactive charts
        
        ### Models Implemented
        
        1. **Cosine Similarity**: Direct cosine similarity between audio features
        2. **K-Nearest Neighbors**: KNN-based similarity search
        3. **FAISS**: Fast similarity search using Facebook's FAISS library
        4. **PCA + Cosine**: Dimensionality reduction followed by cosine similarity
        
        ### Use Cases
        
        - Music recommendation based on acoustic similarity
        - Podcast discovery based on audio characteristics
        - Audiobook recommendations
        - Audio content clustering and organization
        """)


if __name__ == "__main__":
    main()

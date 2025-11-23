"""Audio feature extraction module for recommendation systems."""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import librosa
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class AudioFeatureExtractor:
    """Extract audio features for recommendation systems."""

    def __init__(
        self,
        sample_rate: int = 22050,
        n_mfcc: int = 13,
        n_mels: int = 128,
        hop_length: int = 512,
        n_fft: int = 2048,
    ) -> None:
        """Initialize audio feature extractor.

        Args:
            sample_rate: Target sample rate for audio loading
            n_mfcc: Number of MFCC coefficients to extract
            n_mels: Number of mel bands for spectral features
            hop_length: Number of samples between successive frames
            n_fft: Length of the windowed signal after padding with zeros
        """
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.n_mels = n_mels
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.scaler = StandardScaler()

    def extract_mfcc_features(
        self, audio_path: Union[str, Path], aggregate: str = "mean"
    ) -> np.ndarray:
        """Extract MFCC features from audio file.

        Args:
            audio_path: Path to audio file
            aggregate: How to aggregate features across time ('mean', 'std', 'max', 'min')

        Returns:
            Aggregated MFCC features
        """
        try:
            y, sr = librosa.load(audio_path, sr=self.sample_rate)
            mfcc = librosa.feature.mfcc(
                y=y, sr=sr, n_mfcc=self.n_mfcc, hop_length=self.hop_length
            )

            if aggregate == "mean":
                return np.mean(mfcc, axis=1)
            elif aggregate == "std":
                return np.std(mfcc, axis=1)
            elif aggregate == "max":
                return np.max(mfcc, axis=1)
            elif aggregate == "min":
                return np.min(mfcc, axis=1)
            else:
                raise ValueError(f"Unknown aggregation method: {aggregate}")

        except Exception as e:
            logger.error(f"Error extracting MFCC features from {audio_path}: {e}")
            return np.zeros(self.n_mfcc)

    def extract_spectral_features(
        self, audio_path: Union[str, Path], aggregate: str = "mean"
    ) -> np.ndarray:
        """Extract spectral features from audio file.

        Args:
            audio_path: Path to audio file
            aggregate: How to aggregate features across time

        Returns:
            Aggregated spectral features
        """
        try:
            y, sr = librosa.load(audio_path, sr=self.sample_rate)

            # Spectral centroid
            spectral_centroids = librosa.feature.spectral_centroid(
                y=y, sr=sr, hop_length=self.hop_length
            )

            # Spectral rolloff
            spectral_rolloff = librosa.feature.spectral_rolloff(
                y=y, sr=sr, hop_length=self.hop_length
            )

            # Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(y, hop_length=self.hop_length)

            # Chroma features
            chroma = librosa.feature.chroma_stft(
                y=y, sr=sr, hop_length=self.hop_length
            )

            features = np.concatenate([
                np.mean(spectral_centroids, axis=1),
                np.mean(spectral_rolloff, axis=1),
                np.mean(zcr, axis=1),
                np.mean(chroma, axis=1),
            ])

            return features

        except Exception as e:
            logger.error(f"Error extracting spectral features from {audio_path}: {e}")
            return np.zeros(15)  # Default size for spectral features

    def extract_rhythm_features(
        self, audio_path: Union[str, Path]
    ) -> np.ndarray:
        """Extract rhythm and tempo features from audio file.

        Args:
            audio_path: Path to audio file

        Returns:
            Rhythm features including tempo and beat strength
        """
        try:
            y, sr = librosa.load(audio_path, sr=self.sample_rate)

            # Tempo estimation
            tempo, beats = librosa.beat.beat_track(y=y, sr=sr)

            # Beat strength
            beat_strength = np.mean(librosa.beat.beat_strength(y=y, sr=sr))

            # Rhythm regularity
            rhythm_regularity = np.std(np.diff(beats)) if len(beats) > 1 else 0

            return np.array([tempo, beat_strength, rhythm_regularity])

        except Exception as e:
            logger.error(f"Error extracting rhythm features from {audio_path}: {e}")
            return np.zeros(3)

    def extract_all_features(
        self, audio_path: Union[str, Path]
    ) -> np.ndarray:
        """Extract comprehensive audio features.

        Args:
            audio_path: Path to audio file

        Returns:
            Combined feature vector
        """
        mfcc_features = self.extract_mfcc_features(audio_path)
        spectral_features = self.extract_spectral_features(audio_path)
        rhythm_features = self.extract_rhythm_features(audio_path)

        return np.concatenate([mfcc_features, spectral_features, rhythm_features])

    def extract_features_batch(
        self, audio_paths: List[Union[str, Path]]
    ) -> Tuple[np.ndarray, List[str]]:
        """Extract features for multiple audio files.

        Args:
            audio_paths: List of paths to audio files

        Returns:
            Feature matrix and list of valid file paths
        """
        features_list = []
        valid_paths = []

        for path in audio_paths:
            try:
                features = self.extract_all_features(path)
                features_list.append(features)
                valid_paths.append(str(path))
            except Exception as e:
                logger.warning(f"Skipping {path} due to error: {e}")

        if not features_list:
            raise ValueError("No valid audio files found")

        return np.array(features_list), valid_paths

    def fit_scaler(self, features: np.ndarray) -> None:
        """Fit the feature scaler.

        Args:
            features: Feature matrix to fit scaler on
        """
        self.scaler.fit(features)

    def transform_features(self, features: np.ndarray) -> np.ndarray:
        """Transform features using fitted scaler.

        Args:
            features: Feature matrix to transform

        Returns:
            Scaled feature matrix
        """
        return self.scaler.transform(features)

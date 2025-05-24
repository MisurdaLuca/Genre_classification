import streamlit as st
import os
import librosa
import numpy as np
import pandas as pd
import joblib
from scipy.stats import skew, kurtosis
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
import tempfile

# Set page configuration
st.set_page_config(page_title="Genrify", page_icon="🎵")

st.title("Genrify: Music Genre Analysis App")
st.markdown("Upload an audio file to split it into 20-second segments, extract features from each, average them, and predict genres.")

UPLOAD_DIR = "uploaded_files"
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

NUMBER_OF_MFCC = 13

def extract_band_features(signal, sr, freq_ranges):
    stft = librosa.stft(signal)
    freqs = librosa.fft_frequencies(sr=sr)
    mag = np.abs(stft)

    band_features = {}
    for i, (low, high) in enumerate(freq_ranges):
        band_idx = np.where((freqs >= low) & (freqs < high))[0]
        if band_idx.size == 0:
            continue
        band_mag = mag[band_idx, :]

        band_energy = np.mean(np.sum(band_mag**2, axis=0))
        band_rms = np.mean(np.sqrt(np.mean(band_mag**2, axis=0)))
        band_centroid = np.mean(librosa.feature.spectral_centroid(S=band_mag, sr=sr)[0])

        band_mean = np.mean(band_mag)
        band_std = np.std(band_mag)
        band_skew = skew(band_mag.flatten())
        band_kurtosis = kurtosis(band_mag.flatten())

        band_features.update({
            f"band_{i}_label": i,
            f"band_{i}_energy": band_energy,
            f"band_{i}_rms": band_rms,
            f"band_{i}_centroid": band_centroid,
            f"band_{i}_mean": band_mean,
            f"band_{i}_std": band_std,
            f"band_{i}_skew": band_skew,
            f"band_{i}_kurtosis": band_kurtosis,
        })
    return band_features

def extract_chromogram_features(signal, sr):
    chroma = librosa.feature.chroma_stft(y=signal, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)
    chroma_std = np.std(chroma, axis=1)
    chroma_skew = skew(chroma, axis=1)
    chroma_kurtosis = kurtosis(chroma, axis=1)

    chromogram_features = {}
    for i in range(chroma.shape[0]):
        chromogram_features[f"chroma_{i}_mean"] = chroma_mean[i]
        chromogram_features[f"chroma_{i}_std"] = chroma_std[i]
        chromogram_features[f"chroma_{i}_skew"] = chroma_skew[i]
        chromogram_features[f"chroma_{i}_kurtosis"] = chroma_kurtosis[i]
    return chromogram_features

def extract_perceptual_shockwave_by_band(signal, sr, freq_ranges):
    stft = librosa.stft(signal)
    mag = np.abs(stft)
    freqs = librosa.fft_frequencies(sr=sr)

    shockwave_features = {}
    for i, (low, high) in enumerate(freq_ranges):
        band_idx = np.where((freqs >= low) & (freqs < high))[0]
        if band_idx.size == 0:
            continue

        band_mag = mag[band_idx, :]
        band_signal = np.sum(band_mag, axis=0)
        band_signal = band_signal / np.max(np.abs(band_signal)) if np.max(np.abs(band_signal)) > 0 else band_signal

        onset_env = librosa.onset.onset_strength(S=band_mag, sr=sr)
        shockwave_features[f"shock_band_{i}_onset_mean"] = np.mean(onset_env)
        shockwave_features[f"shock_band_{i}_onset_std"] = np.std(onset_env)
        shockwave_features[f"shock_band_{i}_onset_skew"] = skew(onset_env)
        shockwave_features[f"shock_band_{i}_onset_kurtosis"] = kurtosis(onset_env)

        flux = np.sqrt(np.sum(np.diff(band_mag, axis=1)**2, axis=0))
        shockwave_features[f"shock_band_{i}_flux_mean"] = np.mean(flux)
        shockwave_features[f"shock_band_{i}_flux_std"] = np.std(flux)
        shockwave_features[f"shock_band_{i}_flux_skew"] = skew(flux)
        shockwave_features[f"shock_band_{i}_flux_kurtosis"] = kurtosis(flux)
    return shockwave_features

def extract_features_from_segment(signal, sr, freq_ranges):
    features = {}

    chroma_feats = extract_chromogram_features(signal, sr)
    features.update(chroma_feats)

    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=NUMBER_OF_MFCC)
    for i in range(NUMBER_OF_MFCC):
        features[f"mfcc_{i}_mean"] = np.mean(mfcc[i])
        features[f"mfcc_{i}_std"] = np.std(mfcc[i])

    band_feats = extract_band_features(signal, sr, freq_ranges)
    features.update(band_feats)

    shockwave_feats = extract_perceptual_shockwave_by_band(signal, sr, freq_ranges)
    features.update(shockwave_feats)

    zcr = librosa.feature.zero_crossing_rate(signal)[0]
    features["zero_crossing_rate_mean"] = np.mean(zcr)
    features["zero_crossing_rate_std"] = np.std(zcr)

    spectrogram_db = librosa.amplitude_to_db(np.abs(librosa.stft(signal)), ref=np.max)
    features["spectrogram_mean"] = np.mean(spectrogram_db)
    features["spectrogram_std"] = np.std(spectrogram_db)

    mel_db = librosa.amplitude_to_db(librosa.feature.melspectrogram(y=signal, sr=sr), ref=np.max)
    features["mel_spectrogram_mean"] = np.mean(mel_db)
    features["mel_spectrogram_std"] = np.std(mel_db)

    harmonic, percussive = librosa.effects.hpss(signal)
    features["harmonics_mean"] = np.mean(harmonic)
    features["harmonics_std"] = np.std(harmonic)
    features["percussive_mean"] = np.mean(percussive)
    features["percussive_std"] = np.std(percussive)

    centroid = librosa.feature.spectral_centroid(y=signal, sr=sr)[0]
    features["spectral_centroids_mean"] = np.mean(centroid)
    features["spectral_centroids_std"] = np.std(centroid)
    features["spectral_centroid_skew"] = skew(centroid)
    features["spectral_centroid_kurtosis"] = kurtosis(centroid)

    tempo = librosa.beat.beat_track(y=signal, sr=sr)[0]
    features["tempo_bpm"] = tempo.item() if tempo.size > 0 else 0

    rolloff = librosa.feature.spectral_rolloff(y=signal, sr=sr)[0]
    features["spectral_rolloff_mean"] = np.mean(rolloff)
    features["spectral_rolloff_std"] = np.std(rolloff)

    bandwidth = librosa.feature.spectral_bandwidth(y=signal, sr=sr)[0]
    features["spectral_bandwidth_mean"] = np.mean(bandwidth)
    features["spectral_bandwidth_std"] = np.std(bandwidth)

    features["rms_energy_mean"] = np.mean(librosa.feature.rms(y=signal)[0])
    features["rms_energy_std"] = np.std(librosa.feature.rms(y=signal)[0])

    nan_features = {k: v for k, v in features.items() if isinstance(v, (int, float)) and np.isnan(v)}
    if nan_features:
        st.warning(f"NaN values detected in features: {nan_features}")

    return features

def process_audio_multisegment(audio_path, segment_length_sec=20):
    try:
        y, sr = librosa.load(audio_path)
        signal, _ = librosa.effects.trim(y)
        audio_length = librosa.get_duration(y=signal, sr=sr)

        segments = []
        segment_samples = int(segment_length_sec * sr)
        total_samples = len(signal)

        freq_ranges = [
            (20, 60), (60, 120), (120, 250), (250, 500), (500, 1000),
            (1000, 2000), (2000, 4000), (4000, 8000), (8000, 12000),
            (12000, 16000), (16000, 20000)
        ]

        for start in range(0, total_samples, segment_samples):
            end = min(start + segment_samples, total_samples)
            segment_signal = signal[start:end]
            features = extract_features_from_segment(segment_signal, sr, freq_ranges)
            segments.append(features)

        for i, segment in enumerate(segments):
            non_scalar_features = {k: v for k, v in segment.items() if isinstance(v, (np.ndarray, list))}
            if non_scalar_features:
                st.warning(f"Non-scalar values in segment {i}: {non_scalar_features}")

        df_segments = pd.DataFrame(segments)
        mean_features = df_segments.mean(axis=0).to_dict()

        mean_features["file_name"] = audio_path
        mean_features["audio_length"] = audio_length
        mean_features["num_segments"] = len(segments)

        return mean_features, segments
    except Exception as e:
        st.error(f"Error processing audio segments: {str(e)}")
        return None, None

def predict_top_3_genres(model, X, label_encoder):
    try:
        probas = model.predict_proba(X)
        top_3_indices = np.argsort(probas, axis=1)[:, -3:][:, ::-1]
        top_3_probas = np.sort(probas, axis=1)[:, -3:][:, ::-1]
        top_3_labels = [label_encoder.inverse_transform(indices) for indices in top_3_indices]
        return [(labels, probas) for labels, probas in zip(top_3_labels, top_3_probas)]
    except Exception as e:
        st.error(f"Error predicting genres: {str(e)}")
        return []

# Load model and encoder
try:
    knn_model = joblib.load("knn_weighted.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
except FileNotFoundError:
    st.error("Model or label encoder file not found. Please ensure 'knn_model.pkl' and 'label_encoder.pkl' are available.")
    st.stop()

uploaded_file = st.file_uploader("Upload an audio file (wav, mp3, ogg, flac, m4a)", type=["wav", "mp3", "ogg", "flac", "m4a"])

if uploaded_file is not None:
    try:
        temp_file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        mean_features, segments = process_audio_multisegment(temp_file_path, segment_length_sec=20)

        if mean_features is not None and segments is not None:
            st.write(f"Number of extracted segments: {mean_features['num_segments']}")

            # Define freq_ranges before using it
            freq_ranges = [
                (20, 60), (60, 120), (120, 250), (250, 500), (500, 1000),
                (1000, 2000), (2000, 4000), (4000, 8000), (8000, 12000),
                (12000, 16000), (16000, 20000)
            ]

            # Segment-wise predictions
            st.subheader("Genre Predictions for Each Segment")
            features_df = pd.DataFrame(segments)
            exclude_columns = ['file_name', 'audio_length', 'num_segments'] + [f"band_{i}_range" for i in range(len(freq_ranges))]
            prediction_features = [col for col in features_df.columns if col not in exclude_columns]

            if len(prediction_features) != 239:
                st.error(f"Extracted {len(prediction_features)} features per segment, but model expects 239.")
                st.stop()

            X_segments = features_df[prediction_features].values
            imputer = SimpleImputer(strategy="mean")
            X_segments = imputer.fit_transform(X_segments)
            scaler = StandardScaler()
            X_segments = scaler.fit_transform(X_segments)

            segment_predictions = predict_top_3_genres(knn_model, X_segments, label_encoder)

            # Collect maximum probabilities for each genre
            genre_max_probs = {}
            if segment_predictions:
                for i, (labels, probs) in enumerate(segment_predictions):
                    st.write(f"**Segment {i+1} (Time: {i*20}-{(i+1)*20}s)**")
                    for genre, prob in zip(labels, probs):
                        st.write(f"- {genre}: {prob:.2%}")
                        # Update max probability for the genre
                        if genre not in genre_max_probs or prob > genre_max_probs[genre][0]:
                            genre_max_probs[genre] = (prob, i+1)
                    st.write("---")

                # Display top 3 genres with highest probabilities
                st.subheader("Top 3 Genres with Highest Probabilities Across All Segments")
                top_genres = sorted(genre_max_probs.items(), key=lambda x: x[1][0], reverse=True)[:3]
                for genre, (prob, segment_idx) in top_genres:
                    st.write(f"- {genre}: {prob:.2%} (Segment {segment_idx}, Time: {(segment_idx-1)*20}-{segment_idx*20}s)")
            else:
                st.warning("No predictions returned for segments. Check model and input data.")

            # Average features and predictions
            features_df = pd.DataFrame([mean_features])

            for i, (low, high) in enumerate(freq_ranges):
                if f"band_{i}_label" in features_df.columns:
                    features_df[f"band_{i}_range"] = f"{low}-{high}Hz"

            st.markdown("""
            ### Extracted feature types:
            - Chroma features: 12 chromatic components statistics
            - MFCC: 13 Mel-frequency cepstral coefficients statistics
            - Frequency band features: 11 frequency bands with energy, RMS, centroid, mean, std, skew, kurtosis
            - Shockwave features: onset and flux stats per band
            - Zero crossing rate, Spectrogram, Mel spectrogram, Harmonics, Percussive, Spectral centroid, Tempo, etc.
            """)

            st.subheader("Average feature values over all segments")
            st.write(features_df.T)

            exclude_columns = ['file_name', 'audio_length', 'num_segments'] + [f"band_{i}_range" for i in range(len(freq_ranges))]
            prediction_features = [col for col in features_df.columns if col not in exclude_columns]

            if len(prediction_features) != 239:
                st.error(f"Extracted {len(prediction_features)} features, but model expects 239.")
                st.stop()

            X = features_df[prediction_features].values
            imputer = SimpleImputer(strategy="mean")
            X = imputer.fit_transform(X)
            scaler = StandardScaler()
            X = scaler.fit_transform(X)

            st.write("Scaled Feature Vector Shape (Average):", X.shape)
            st.write("Genre Classes:", label_encoder.classes_)

        else:
            st.warning("Failed to extract features from the audio.")

    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
else:
    st.info("Please upload an audio file to start analysis.")
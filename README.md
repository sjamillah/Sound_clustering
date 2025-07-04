# Audio Data Clustering: Unsupervised Learning for Sound Analysis

A machine learning project that automatically discovers patterns in unlabeled audio data using clustering techniques.

## Overview

This project tackles the challenge of organizing and understanding audio data without any prior labels or categories. Instead of working with clean, pre-categorized sound files, we're dealing with raw audio - the kind you might find in a messy music collection, field recordings, or environmental sound datasets where you don't know what you're looking at beforehand.

The approach mimics how humans naturally group sounds - by recognizing similarities in rhythm, pitch, texture, and other audio characteristics. This has practical applications in organizing large audio libraries, categorizing environmental sounds, analyzing speech patterns, or discovering structure in any collection of audio files.

## What This Project Does

The pipeline transforms raw audio waveforms into numerical features that capture the essential characteristics of each sound. We then use dimensionality reduction to make this high-dimensional data manageable and apply clustering algorithms to find natural groupings in the data.

Key components:
- Extract meaningful features from audio files using MFCC (Mel-Frequency Cepstral Coefficients)
- Reduce dimensionality using PCA and t-SNE for visualization and improved clustering
- Apply K-Means and DBSCAN clustering to discover natural sound groups
- Evaluate clustering quality using standard metrics
- Visualize results to understand the discovered patterns

## Getting Started

### Requirements

You'll need Python 3.7+ and these libraries:
```
librosa
scikit-learn
pandas
numpy
matplotlib
seaborn
```

For Google Colab (recommended for beginners):
```python
pip install librosa scikit-learn pandas numpy matplotlib seaborn
```

### Basic Usage

```python
# Load the required functions
from audio_data_clustering import *

# Extract features from your audio files
audio_files = [list of your .wav/.mp3 file paths]
features_df = extract_features(audio_files)

# Reduce dimensions for better clustering
features_pca, pca_model = apply_dimensionality_reduction(features_df, method='pca', n_components=3)

# Find clusters
labels, clustering_model = perform_clustering(features_pca, method='kmeans', n_clusters=3)

# See how well it worked
results = evaluate_clustering(features_pca, labels)
```

## How It Works

### Feature Extraction with MFCC

Rather than working with raw audio waveforms (which are just long sequences of amplitude values), we extract MFCC features. These are particularly good for audio analysis because:

- They focus on frequencies that matter most to human hearing
- They compress the audio information into a manageable number of features (13 coefficients)
- They're robust to noise and recording variations
- They capture both the spectral shape and how it changes over time

For each audio file, we calculate 13 MFCC coefficients and then compute the mean and standard deviation of these coefficients across the entire audio clip. This gives us 26 features per audio file that capture both the average characteristics and the variability of the sound.

### Dimensionality Reduction

Working with 26-dimensional data directly is challenging for visualization and can hurt clustering performance. We use two approaches:

**PCA (Principal Component Analysis)**: This finds the directions in the data that capture the most variance. It's great for understanding which features matter most and reduces computational complexity while preserving the global structure of the data.

**t-SNE**: This is better at preserving local neighborhoods and often reveals cluster structure that PCA misses. It's particularly good for visualization because it tends to create tight, well-separated groups.

### Clustering Algorithms

**K-Means**: Works well when you expect roughly spherical clusters of similar sizes. We use the elbow method to automatically determine the optimal number of clusters by looking for the point where adding more clusters doesn't significantly improve the within-cluster sum of squares.

**DBSCAN**: Doesn't require you to specify the number of clusters beforehand and can find irregularly shaped clusters. It also automatically identifies noise points that don't belong to any cluster. We tune the epsilon parameter using k-distance graphs.

## Results and Findings

Based on our experiments with unlabeled audio data:

### Performance Comparison

| Method | Silhouette Score | Davies-Bouldin Index | Notes |
|--------|------------------|---------------------|-------|
| K-Means on PCA | 0.3713 | ~1.0 | Best overall performance |
| DBSCAN on PCA | -0.3384 | 1.44 | Struggled with this dataset |

### Key Insights

**Dimensionality reduction was crucial**: The original 26-dimensional feature space was too complex for effective clustering. PCA helped by:
- Concentrating most of the meaningful variation into just 3 dimensions
- Filtering out noise and redundant information
- Making the clusters more compact and separable

**K-Means worked better than DBSCAN for this data**: This was somewhat surprising since DBSCAN is often praised for handling irregular cluster shapes. However:
- The audio features formed relatively spherical clusters in the PCA space
- DBSCAN was sensitive to parameter choices and classified too many points as noise
- K-Means' assumption of roughly equal-sized, spherical clusters matched our data well

**PCA vs t-SNE for visualization**: While both techniques reduced dimensionality effectively, PCA provided clearer, more distinct clusters. t-SNE sometimes created scattered or overlapping groups that were harder to interpret, though it did reveal some interesting local structure.

## Challenges and Limitations

**Parameter sensitivity**: Both DBSCAN's epsilon parameter and the choice of number of components for dimensionality reduction significantly affected results. We spent considerable time tuning these.

**Audio quality variations**: Real-world audio files have different recording qualities, lengths, and background noise levels. The feature extraction process handles this reasonably well, but very noisy or very short clips can still be problematic.

**Interpreting clusters**: Without ground truth labels, it's challenging to know if the discovered clusters correspond to meaningful audio categories. Visual and auditory inspection of cluster members is necessary to validate results.

**Computational requirements**: t-SNE in particular can be slow on larger datasets, and the feature extraction process requires loading entire audio files into memory.

## File Organization

```
Audio_Data_Clustering.ipynb
README.md                   
data/
```

## Real-World Applications

This type of unsupervised audio analysis is useful for:

- **Music library organization**: Automatically grouping songs by genre, mood, or style
- **Environmental sound analysis**: Categorizing urban noise, nature sounds, or industrial audio
- **Speech analysis**: Finding patterns in recorded conversations or lectures
- **Audio quality control**: Identifying recording problems or unusual audio in large datasets
- **Sound effect libraries**: Organizing large collections of sound effects for media production

## Future Improvements

Several directions could improve this work:

- **Additional features**: Spectral centroid, zero-crossing rate, or chroma features might capture different aspects of audio
- **Deep learning features**: Pre-trained audio neural networks could provide richer representations
- **Temporal modeling**: Current approach treats each audio file as a single point; modeling temporal sequences could be valuable
- **Interactive visualization**: Tools for listening to cluster members and understanding what makes them similar
- **Scalability**: Optimizations for handling thousands of audio files efficiently

## Contributing

If you'd like to improve this project:

1. Try it with your own audio datasets and share what you find
2. Experiment with different features or clustering algorithms
3. Improve the visualization or add interactive elements
4. Add better error handling or support for more audio formats
5. Create examples with specific types of audio (music, speech, environmental sounds)

## Technical Notes

The code is designed to work in Google Colab but can be adapted for local use. The main dependencies are librosa for audio processing and scikit-learn for machine learning. All visualizations use matplotlib and seaborn.

For reproducibility, random seeds are set where possible, though t-SNE can still show some variation between runs.

The feature extraction process assumes mono audio and automatically resamples to a consistent sample rate. Very short audio clips (less than a few seconds) may not provide reliable features.

## License

MIT License - feel free to use this code for your own projects, academic work, or commercial applications.

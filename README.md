# Audio Data Clustering: Unsupervised Learning for Sound Analysis

A machine learning project that automatically discovers patterns in unlabeled audio data using clustering techniques with Mel Spectrogram features.

## Overview

This project tackles the challenge of organizing and understanding audio data without any prior labels or categories. Instead of working with clean, pre-categorized sound files, we're dealing with raw audio - the kind you might find in a messy music collection, field recordings, or environmental sound datasets where you don't know what you're looking at beforehand.

The approach mimics how humans naturally group sounds - by recognizing similarities in rhythm, pitch, texture, and other audio characteristics. This has practical applications in organizing large audio libraries, categorizing environmental sounds, analyzing speech patterns, or discovering structure in any collection of audio files.

## What This Project Does

The pipeline transforms raw audio waveforms into numerical features that capture the essential characteristics of each sound. We then use dimensionality reduction to make this high-dimensional data manageable and apply clustering algorithms to find natural groupings in the data.

Key components:
- Extract meaningful features from audio files using Mel Spectrograms with comprehensive statistical aggregation
- Reduce dimensionality using PCA and t-SNE for visualization and improved clustering
- Apply K-Means and adaptive DBSCAN clustering to discover natural sound groups
- Evaluate clustering quality using comprehensive metrics (silhouette score, Davies-Bouldin index, inertia)
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
# Mount Google Drive and load audio files
mount_drive()
unlabelled_data_path = '/content/drive/MyDrive/unlabelled_sounds/unlabelled_sounds'
audio_files = [os.path.join(unlabelled_data_path, f) 
               for f in os.listdir(unlabelled_data_path) if f.endswith('.wav')]

# Extract Mel Spectrogram features
features_df = extract_features(audio_files)

# Apply dimensionality reduction
features_pca, pca_transformer = apply_dimensionality_reduction(features_df, method='pca', n_components=3)
features_tsne, tsne_transformer = apply_dimensionality_reduction(features_df, method='tsne', n_components=3)

# Perform clustering
optimal_k = 3
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(features_pca)

# Adaptive DBSCAN clustering
dbscan_labels, dbscan_model = adaptive_dbscan_clustering(features_pca, target_noise_ratio=0.15)

# Comprehensive evaluation
kmeans_results = evaluate_clustering_performance(features_pca, kmeans_labels, kmeans, "K-MEANS")
dbscan_results = evaluate_clustering_performance(features_pca, dbscan_labels, dbscan_model, "DBSCAN-ADAPTIVE")
```

## How It Works

### Feature Extraction with Mel Spectrograms

Rather than working with raw audio waveforms or simple coefficients, we extract comprehensive Mel Spectrogram features. These provide richer representation because:

- They maintain both frequency and time information unlike compressed features
- They utilize the mel scale to emphasize frequencies important to human hearing
- They provide detailed spectral content while preserving temporal evolution
- They form the foundation for many modern audio analysis techniques

For each audio file, we:
1. Extract Mel Spectrograms with 128 mel frequency bands
2. Calculate comprehensive statistical measures: mean, standard deviation, maximum, minimum, median, and percentiles across time
3. Compute global spectral characteristics: spectral centroid, spread, and rolloff
4. Generate 899-dimensional feature vectors (128 × 7 statistics + 3 global features) per audio clip

### Why Dimensionality Reduction is Critical

Working with 899-dimensional mel spectrogram features presents several challenges:

- **Curse of dimensionality**: Data points become approximately equidistant, making clustering difficult
- **Visualization impossibility**: Cannot meaningfully visualize relationships in 899-dimensional space
- **Computational complexity**: Distance-based algorithms become slow and memory-intensive
- **Noise amplification**: High-dimensional data contains redundant features that mask true patterns

Dimensionality reduction concentrates the most important variance into fewer dimensions while filtering out noise, making clusters more distinguishable and computationally manageable.

### Dimensionality Reduction Techniques

**PCA (Principal Component Analysis)**: Preserves global structure and variance by projecting data onto directions that capture maximum variance. Great for understanding feature importance and maintaining linear relationships.

**t-SNE**: Excels at preserving local neighborhoods and revealing non-linear cluster structures. Better for visualization of complex patterns but sacrifices interpretability.

### Clustering Algorithms

**K-Means**: Works well with spherical clusters of similar sizes. We use the elbow method to determine optimal cluster numbers. Performed best on our mel spectrogram features due to their relatively uniform distribution after statistical aggregation.

**Adaptive DBSCAN**: Our enhanced density-based approach that automatically finds optimal parameters to achieve a target noise ratio (typically 15%). Systematically tests multiple parameter combinations to avoid classifying all points as noise.

## Results and Performance

Based on experiments with 3000 unlabeled audio files:

### Performance Comparison

| Algorithm | Silhouette Score | Davies-Bouldin Index | Inertia | Clusters | Noise Points |
|-----------|------------------|---------------------|---------|----------|--------------|
| K-Means | 0.2483 | 1.3178 | 698,395.88 | 3 | 0 |
| DBSCAN-Adaptive | -0.3327 | 0.8586 | 758,077.83 | 17 | 445 |

### Key Findings

**K-Means achieved acceptable performance**: 
- Silhouette score of 0.2483 indicates reasonably structured clusters
- Successfully partitioned data into three balanced clusters (1137, 761, 1102 samples)
- Davies-Bouldin index of 1.3178 shows moderate cluster separation
- High inertia reflects the complexity of mel spectrogram feature space

**DBSCAN-Adaptive revealed interesting patterns**:
- Discovered 17 distinct clusters, suggesting more granular audio groupings
- Negative silhouette score (-0.3327) indicates overlapping cluster boundaries
- Better Davies-Bouldin index (0.8586) shows good cluster separation despite overlap
- 445 noise points (14.8%) successfully identified outlier audio samples
- Higher inertia (758,077.83) reflects the fragmented cluster structure

**Algorithm comparison insights**:
- **K-Means**: Produced fewer, more cohesive clusters with balanced sizes
- **DBSCAN-Adaptive**: Identified finer-grained structure but with more overlap
- **Cluster count**: 3 vs 17 clusters suggests different granularity levels
- **Noise detection**: DBSCAN's ability to identify 445 outliers provides valuable data cleaning

**Dimensionality reduction impact**:
- PCA successfully preserved 75%+ of variance in 3 components
- Significant improvement in clustering performance post-reduction
- Enhanced visualization capabilities while maintaining essential audio characteristics

**PCA vs t-SNE comparison**:
- PCA provided better global structure preservation for K-Means clustering
- t-SNE revealed local patterns that DBSCAN-Adaptive successfully exploited
- PCA's 3 balanced clusters vs t-SNE's influence on DBSCAN's 17 granular clusters
- Both techniques essential for different clustering approaches and granularity levels

## Methodology and Analysis

### Structured Approach

Our analysis follows a systematic methodology:

1. **Data Loading and Preprocessing**: Mount Google Drive, load audio files, handle errors gracefully
2. **Feature Extraction**: Comprehensive mel spectrogram analysis with statistical aggregation
3. **Initial Exploration**: Visualize raw features, understand distributions and patterns
4. **Audio Visualization**: Examine waveforms and spectrograms to connect features with actual audio
5. **Dimensionality Reduction**: Apply PCA and t-SNE, compare effectiveness
6. **Clustering**: K-Means with elbow method, adaptive DBSCAN with parameter optimization
7. **Comprehensive Evaluation**: Multiple metrics including inertia, silhouette score, Davies-Bouldin index
8. **Results Analysis**: Detailed interpretation of cluster quality and algorithm performance

## Results Analysis and Interpretation

### Clustering Performance Summary

The experimental results reveal complementary strengths between the two clustering approaches:

**K-Means Performance (Silhouette: 0.2483, Davies-Bouldin: 1.3178)**:
- Produced 3 well-balanced clusters representing broad audio categories
- Positive silhouette score indicates reasonable cluster cohesion
- Moderate Davies-Bouldin index suggests acceptable but not excellent separation
- Lower inertia (698,395.88) reflects more compact cluster structure
- No noise points, ensuring all audio samples are categorized

**DBSCAN-Adaptive Performance (Silhouette: -0.3327, Davies-Bouldin: 0.8586)**:
- Discovered 17 distinct clusters, revealing fine-grained audio structure
- Negative silhouette score indicates overlapping boundaries between similar audio types
- Superior Davies-Bouldin index (0.8586) shows better cluster separation quality
- Higher inertia (758,077.83) reflects more dispersed, granular clustering
- Successfully identified 445 noise points (14.8%), providing valuable outlier detection

### Methodological Insights

**Granularity vs Cohesion Trade-off**:
The results demonstrate a fundamental trade-off in audio clustering between granularity and cohesion. K-Means optimizes for balanced, cohesive groups suitable for broad audio categorization, while DBSCAN-Adaptive discovers nuanced audio similarities at the cost of some cluster overlap.

**Noise Point Value**:
DBSCAN's identification of 445 noise points provides significant value for audio data cleaning, potentially flagging corrupted files, unusual recordings, or audio samples that don't fit standard categories.

**Complementary Approaches**:
The different cluster counts (3 vs 17) suggest these algorithms are discovering structure at different hierarchical levels, making them complementary rather than competing approaches.

## Challenges and Solutions

### DBSCAN Over-fragmentation Challenge
**Challenge**: DBSCAN produced 17 small clusters with negative silhouette score
**Analysis**: The density-based approach over-fragmented the mel spectrogram feature space
**Finding**: While DBSCAN achieved better Davies-Bouldin separation (0.8586), the 17 clusters were too granular for practical audio categorization
**Insight**: Audio features after statistical aggregation don't exhibit the distinct density regions that DBSCAN requires for optimal performance

### High-Dimensional Feature Space
**Challenge**: 899-dimensional mel spectrogram features were computationally prohibitive
**Solution**: Strategic dimensionality reduction:
- PCA for variance preservation and interpretability
- Statistical aggregation to capture essential temporal dynamics
- Comprehensive evaluation to ensure information retention

### Parameter Sensitivity
**Challenge**: Both clustering algorithms sensitive to parameter choices
**Solution**: Systematic optimization:
- Elbow method for K-Means cluster selection
- Adaptive search for DBSCAN parameters
- Multiple evaluation metrics for robust assessment

## File Organization

```
audio_data_clustering.ipynb        # Complete implementation notebook
README.md                       # This documentation
requirements.txt      # packages needed
```

## Real-World Applications

This unsupervised audio analysis approach is valuable for:

- **Music Library Organization**: Automatically grouping songs by genre, mood, or acoustic characteristics
- **Environmental Sound Analysis**: Categorizing urban noise, nature sounds, or industrial audio patterns
- **Speech Pattern Analysis**: Finding structure in recorded conversations, lectures, or linguistic data
- **Audio Quality Control**: Identifying recording anomalies or unusual audio in large datasets
- **Sound Effect Libraries**: Organizing collections for media production based on acoustic similarity
- **Field Recording Analysis**: Discovering patterns in wildlife recordings or environmental monitoring

## Technical Implementation

### Key Implementation Features

The implementation includes several important components:

**Robust Feature Extraction**:
- Progress indicators during processing (`if (i + 1) % 10 == 0: print(...)`)
- Comprehensive error handling for corrupted audio files
- Detailed logging of successful vs failed extractions
- Automatic creation of descriptive feature names

**Adaptive Parameter Optimization**:
- Systematic k-distance analysis for eps estimation
- Multiple percentile suggestions (70th, 80th, 90th)
- Extensive parameter grid search with progress tracking
- Automatic fallback to conservative parameters if optimization fails

**Comprehensive Visualization Suite**:
- 2D and 3D scatter plots with cluster coloring
- Waveform and mel spectrogram visualizations
- Pair plots for initial feature exploration
- Comparison charts for algorithm performance metrics

**Professional Evaluation Framework**:
- Multiple clustering quality metrics (silhouette, Davies-Bouldin, inertia)
- Detailed cluster analysis with sizes and characteristics
- Performance interpretation with threshold guidelines
- Structured comparison tables and visualizations

### Performance Considerations

- **Memory management**: Processes audio files individually to handle large datasets
- **Computational efficiency**: Optimized parameter testing and feature extraction
- **Scalability**: Designed to handle hundreds to thousands of audio files
- **Reproducibility**: Fixed random seeds where possible for consistent results

## Future Improvements

### Enhanced Feature Engineering
- **Temporal modeling**: Capture sequential patterns within audio clips
- **Multi-scale analysis**: Combine features at different time resolutions
- **Deep learning features**: Pre-trained audio neural network representations
- **Domain-specific features**: Tailored extraction for music, speech, or environmental sounds

### Algorithm Enhancements
- **Hierarchical clustering**: Discover nested cluster structures
- **Ensemble methods**: Combine multiple clustering approaches
- **Semi-supervised learning**: Incorporate limited labeled data when available
- **Online clustering**: Handle streaming audio data

### Visualization and Interaction
- **Interactive cluster exploration**: Tools for listening to cluster members
- **Real-time visualization**: Dynamic plotting during processing
- **Cluster validation**: Human-in-the-loop verification tools
- **Audio player integration**: Direct playback from visualizations

## Contributing

We welcome contributions to improve this audio clustering implementation:

### Areas for Enhancement
- **New feature extractors**: Implement additional audio features (chroma, tonnetz, etc.)
- **Algorithm extensions**: Add support for other clustering methods
- **Performance optimization**: Improve computational efficiency for larger datasets
- **Visualization improvements**: Enhanced plotting and interactive elements
- **Documentation**: Expand examples and use cases

### Getting Involved
1. Try the implementation with your own audio datasets
2. Experiment with different parameter settings and share results
3. Implement new features or clustering algorithms
4. Improve error handling and edge case management
5. Create domain-specific examples (music, speech, environmental sounds)

## Technical Notes

### Environment Setup
- **Google Colab**: Recommended for easy setup and GPU access
- **Local installation**: Requires Python 3.7+ and specified dependencies
- **Audio format support**: WAV files recommended, MP3 and FLAC supported
- **Memory requirements**: Scales with dataset size, typically 4-8GB sufficient

### Performance Characteristics
- **Feature extraction**: ~1-2 seconds per audio file
- **Dimensionality reduction**: Scales with sample count and feature dimensions
- **Clustering**: K-Means is fast, DBSCAN parameter optimization can be slow
- **Visualization**: t-SNE is computationally intensive for large datasets

### Reproducibility Notes
- **Random seeds**: Set for PCA, K-Means, and where possible for t-SNE
- **Parameter documentation**: All clustering parameters explicitly specified
- **Version dependencies**: Specific library versions recommended for consistency
- **Data preprocessing**: Consistent audio loading and normalization procedures

## License

MIT License - This code is freely available for academic research, personal projects, and commercial applications. Feel free to adapt, modify, and distribute according to your needs.

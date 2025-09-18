# Bifrost CVCov Plugin - Technical Brief

## Purpose
A FiftyOne plugin for computer vision test coverage analysis through object-level embedding computation and clustering.

## Core Functionality

### 1. Object Detection Processing
- Processes FiftyOne datasets containing detection annotations
- Extracts individual object crops from detection bounding boxes
- Uses FiftyOne's patches system (`dataset.to_patches("detections")`) with fallback to custom implementation

### 2. Embedding Computation
- **Primary**: FiftyOne zoo models (CLIP ViT-Base-32) for standardized embeddings
- **Fallback**: Custom DINOv2/CLIP models when patches system fails
- Stores embeddings directly on detection objects for persistence

### 3. Clustering Analysis
- HDBSCAN clustering on computed embeddings
- Configurable parameters (min_cluster_size=5, epsilon=0.01)
- Identifies similar objects and outliers across the dataset

### 4. Real-time Progress Tracking
- Custom progress system using dataset metadata storage
- Supports multiple concurrent operations (embedding + clustering)
- Automatic operation discovery and progress polling

## Architecture

### Backend (Python)
- **Operators**: FetchAnnotations, ComputeEmbeddings, ClusterEmbeddings, GetProgress
- **Data Flow**: Dataset → Patches → Embeddings → Clustering → Results
- **Progress System**: Non-modal progress tracking via dataset.info storage

### Frontend (React/TypeScript)
- **Panel Component**: Real-time operation monitoring and results display
- **Progress Display**: Dynamic progress bars for active operations
- **Results Visualization**: Cluster metrics, distribution, and statistics
- **State Management**: Recoil integration with FiftyOne state atoms

## Key Components

### Data Processing Pipeline
```
FiftyOne Dataset → Detection Patches → Object Embeddings → HDBSCAN Clusters
```

### User Interface
- Single-button workflow: "Compute Embeddings & Cluster"
- Auto-chaining: Embeddings completion triggers clustering
- Results dashboard: Cluster count, object distribution, noise analysis

## Technical Integration
- **FiftyOne Plugin Framework**: Operators and React panels
- **State Synchronization**: Real-time dataset state monitoring
- **Error Handling**: Graceful fallbacks and user feedback
- **Persistence**: All computed data stored in FiftyOne dataset structure

## Use Case
Enables computer vision teams to analyze test dataset coverage by identifying clusters of similar objects and detecting edge cases (noise points) in their detection datasets.



DATASET PLAN

# Use RWF-2000 as primary dataset (cleanest for fighting)
# Augment with "Fighting" class from Anomaly dataset
# Use Violence detection for additional negative samples

Positive samples:
- RWF-2000 "Fight"
- Anomaly "Fighting" (class 7)

Negative samples:
- RWF-2000 "NonFight"
- Anomaly "Normal_Videos_event" (class 8)
- Violence detection "nonviolent"



GENERAL MODEL PLAN

# Frame-level feature extraction (Option C)
Backbone: EfficientNet-B0 (lightweight, good accuracy)
# Temporal modeling (Option B)
Temporal module: Transformer Encoder (better than LSTM for this)
# Output: Binary classification

Model Structure:
1. Frame encoder (EfficientNet-B0 features)
2. Temporal transformer (8 layers, 4 heads)
3. Classification head (sigmoid output)



FEATURE EXTRACTOR

# Frame-level feature extraction (Option C)
Backbone: EfficientNet-B0 (lightweight, good accuracy)
# Temporal modeling (Option B)
Temporal module: Transformer Encoder (better than LSTM for this)
# Output: Binary classification

Model Structure:
1. Frame encoder (EfficientNet-B0 features)
2. Temporal transformer (8 layers, 4 heads)
3. Classification head (sigmoid output)



CORE COMPONENTS

# Core components to build:
1. VideoDataset loader (handles multiple datasets)
2. Frame sampling strategy (uniform vs random)
3. Mixed precision training (fp16 for speed)
4. Real-time augmentation pipeline


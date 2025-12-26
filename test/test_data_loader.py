# test_data_loader.py
from combine_datasets import DataLoaderFactory, FightDetectionDataset


def test_data_loader():
    # Define your dataset paths
    dataset_paths = {
        'rwf': 'C:/Users/DEvA/Videos/Video_For_AI/RWF-2000/train',
        'anomaly': 'C:/Users/DEvA/Videos/Video_For_AI/Anomaly_detection_dataset/train',
        'violence': 'C:/Users/DEvA/Videos/Video_For_AI/violence-detection-datasets-main'
    }

    # Test single dataset
    print("Testing dataset creation...")
    dataset = FightDetectionDataset(
        dataset_paths=dataset_paths,
        clip_length=16,
        frame_rate=10,
        resolution=(224, 224),
        mode='train',
        datasets_to_use=['rwf']  # Start with just RWF for testing
    )

    print(f"Dataset size: {len(dataset)}")

    # Test one sample
    video, label = dataset[0]
    print(f"Video tensor shape: {video.shape}")  # Should be [3, 16, 224, 224]
    print(f"Label: {label}")

    # Test data loader factory
    print("\nTesting data loader factory...")
    train_loader, val_loader = DataLoaderFactory.create_loaders(
        dataset_paths=dataset_paths,
        batch_size=8,  # Small batch for testing
        clip_length=16,
        frame_rate=10,
        resolution=(224, 224)
    )

    # Test one batch
    for batch_idx, (videos, labels) in enumerate(train_loader):
        print(f"Batch {batch_idx}:")
        print(f"  Videos shape: {videos.shape}")  # [8, 3, 16, 224, 224]
        print(f"  Labels shape: {labels.shape}")  # [8]
        print(f"  Labels: {labels}")

        if batch_idx == 1:  # Just test a couple of batches
            break


if __name__ == "__main__":
    test_data_loader()
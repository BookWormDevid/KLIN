from pathlib import Path

import cv2

class Preprocessor:

    def process_klindataset_safe(self,  dataset_path, chunk_size=150):
        """
        Safe version: Creates processed copies in a new folder structure.
        """
        splits = ["Train", "Val", "Test"]
        classes = ["violent", "nonviolent"]

        # Create a new processed dataset folder
        processed_path = Path(dataset_path).parent / "klin_processed"
        processed_path.mkdir(exist_ok=True)

        for split in splits:
            for class_name in classes:
                input_folder = Path(dataset_path) / split / class_name
                output_folder = processed_path / split / class_name

                if not input_folder.exists():
                    continue

                # Create output folder
                output_folder.mkdir(parents=True, exist_ok=True)

                # Get all video files
                video_files = [
                    f
                    for f in input_folder.iterdir()
                    if f.is_file()
                    and f.suffix.lower() in [".mp4", ".avi", ".mov", ".mkv", ".wmv"]
                ]

                # Start chunk counter for this class
                chunk_counter = 0

                # Process all video files
                for video_file in video_files:
                    chunk_counter = self.create_video_chunks_safe(
                        video_file, output_folder, class_name, chunk_size, chunk_counter
                    )


    def create_video_chunks_safe(self,
        video_path, output_dir, class_name, chunk_size, start_chunk_count
    ):
        """
        Create chunks from a video file - safe version with better error handling.
        """
        try:
            cap = cv2.VideoCapture(str(video_path))

            if not cap.isOpened():
                print(f"Error opening video: {video_path}")
                return start_chunk_count

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            print(f"\nSplitting: {video_path.name}")
            print(f"Total frames: {total_frames}, Chunk size: {chunk_size}")

            chunk_count = start_chunk_count
            current_frame = 0
            chunks_created = 0

            while current_frame < total_frames:
                # Calculate frames for this chunk
                end_frame = min(current_frame + chunk_size, total_frames)
                frames_in_chunk = end_frame - current_frame

                # Only create chunk if it has reasonable size
                if frames_in_chunk >= chunk_size * 0.5:
                    output_filename = f"{class_name}_{chunk_count:05d}.avi"
                    output_path = output_dir / output_filename

                    # Create video writer
                    fourcc = cv2.VideoWriter.fourcc(*"XVID")
                    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

                    # Set starting frame
                    cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)

                    # Write frames for this chunk
                    frames_written = 0
                    for _i in range(frames_in_chunk):
                        ret, frame = cap.read()
                        if ret:
                            out.write(frame)
                            frames_written += 1
                        else:
                            break

                    out.release()

                    if frames_written > 0:
                        print(f"Created: {output_filename} ({frames_written} frames)")
                        chunk_count += 1
                        chunks_created += 1
                    else:
                        # Delete empty output file
                        if output_path.exists():
                            output_path.unlink()

                current_frame += chunk_size

            cap.release()
            print(f"Total chunks created from this video: {chunks_created}")

            return chunk_count

        except Exception as e:
            print(f"Error processing {video_path}: {e}")
            return start_chunk_count


    def run(self, dataset_path: Path):
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")

        try:
            print("Starting video chunk processing...")
            self.process_klindataset_safe(dataset_path, chunk_size=150)
            print("\nProcessing complete! Check the klin_processed folder.")
        except KeyboardInterrupt:
            print("\nProcessing interrupted by user!")
        except Exception as e:
            print(f"\nUnexpected error: {e}")
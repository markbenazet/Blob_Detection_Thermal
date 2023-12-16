# Blob Detection Algorithm

This algorithm analyzes live video feed from a thermal camera and applies a blob detection algorithm every second. It determines whether there is an object or blob present in the frame based on the algorithm's return value.

## How it Works

1. The algorithm receives a live video feed from a thermal camera.
2. Every second, the algorithm applies the blob detection algorithm to the current frame.
3. The blob detection algorithm analyzes the frame and determines whether there is an object or blob present.
4. If the algorithm detects a blob, it returns true. Otherwise, it returns false.

## Usage

To use this algorithm, follow these steps:

1. Set up a thermal camera and ensure it is connected to the system.
2. Run the algorithm script.
3. The algorithm will continuously analyze the live video feed and provide the output indicating the presence of a blob.



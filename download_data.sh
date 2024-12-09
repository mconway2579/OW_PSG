#!/bin/bash

# Script to download the SUN RGB-D dataset

# Set the URL for the dataset
DATASET_URL="https://rgbd.cs.princeton.edu/data/SUNRGBD.zip"

# Set the destination file name
DEST_FILE="SUNRGBD.zip"

# Start downloading
echo "Starting download of SUN RGB-D dataset..."
if command -v curl &> /dev/null; then
    # Use curl if available
    curl -o "$DEST_FILE" "$DATASET_URL"
elif command -v wget &> /dev/null; then
    # Use wget if available
    wget -O "$DEST_FILE" "$DATASET_URL"
else
    echo "Error: Neither 'curl' nor 'wget' is installed. Please install one of them and try again."
    exit 1
fi

# Check if the download was successful
if [[ -f "$DEST_FILE" ]]; then
    echo "Download completed successfully: $DEST_FILE"
    echo "Unzipping the dataset..."
    unzip "$DEST_FILE" || { echo "Error: Failed to unzip the dataset."; exit 1; }
    echo "Dataset unzipped successfully."
else
    echo "Error: Download failed."
    exit 1
fi

#!/bin/bash

# PARAMS
ZIP_FILE="ConvFinQA/data.zip"
UNZIP_DIR="convfinqa_data"
PREPROCESS="preprocess_data.py"

# Create data_dir if does not exits
if [ ! -d  "$UNZIP_DIR" ]; then
    
    mkdir "$UNZIP_DIR"
    echo "Created $UNZIP_DIR..."
else
    echo "$UNZIP_DIR already exists..."
fi

# Unzip data to target dir
unzip -d "$UNZIP_DIR" -j "$ZIP_FILE"
echo "Unzipped $ZIP_FILE to $UNZIP_DIR"

# Preprocess to form for huggingfaces data
python "$PREPROCESS"
echo "Preprocessed data now in data/"
echo "DONE"

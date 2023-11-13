#!/bin/bash

# Specify the input and output directories
input_directory="svg"  # Replace with your input directory path
output_directory="png"  # Replace with your output directory path

# Create the output directory if it doesn't exist
mkdir -p "$output_directory"

# Loop through each SVG file in the input directory and convert to PNG
for file in "$input_directory"/*.svg; do
    if [ -f "$file" ]; then
        filename=$(basename -- "$file")
        filename_no_ext="${filename%.*}"
        inkscape --export-type="png" --export-filename="$output_directory/${filename_no_ext}.png" "$file"
        echo "Converted $file to $output_directory/${filename_no_ext}.png"
    fi
done

echo "Conversion complete!"

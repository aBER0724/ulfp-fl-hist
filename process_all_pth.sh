#!/bin/bash

# Navigate to the project directory
cd /Users/aber/Documents/Graduate\ Design/Code/dp-fl-reproduction

# Create output directories if they don't exist
mkdir -p record/creditcard/rdp_corrected
mkdir -p record/mnist/rdp_corrected

# Process each .pth file
for pth_file in $(find record -name "*.pth" -type f | grep -v "(rdp).pth"); do
  # Get the base filename without path
  filename=$(basename "$pth_file")
  
  # Determine the output path based on whether it's mnist or creditcard
  if [[ "$pth_file" == *"/mnist/"* ]]; then
    output_dir="record/mnist/rdp_corrected"
  else
    output_dir="record/creditcard/rdp_corrected"
  fi
  
  # Create output filename with (rdp) suffix before .pth
  output_filename="${filename%.pth}(rdp).pth"
  output_path="$output_dir/$output_filename"
  
  echo "Processing: $pth_file"
  echo "Output to: $output_path"
  
  # Run the fix_epsilon_rdp.py script
  python fix_epsilon_rdp.py "$pth_file" "$output_path"
  
  echo "Done with $filename"
  echo "-----------------------------------------"
done

echo "All files processed!" 
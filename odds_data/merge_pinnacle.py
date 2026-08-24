import pandas as pd
import glob
import os

# Set the path to the folder containing your CSV files. 
# Use '.' if the script is in the same folder as the files.
folder_path = '.'

# Use glob to find all files matching your naming structure
file_pattern = os.path.join(folder_path, 'pinnacle_*.csv')
file_list = glob.glob(file_pattern)

# Create an empty list to store each dataframe
dataframes = []

# Loop through the files, read them, and add them to the list
for file in file_list:
    df = pd.read_csv(file)
    dataframes.append(df)

# Check if any files were found before merging
if dataframes:
    # Concatenate all dataframes in the list into one
    merged_df = pd.concat(dataframes, ignore_index=True)
    
    # Export the merged dataframe to a new CSV file
    output_filename = 'pinnacle_merged_all.csv'
    merged_df.to_csv(output_filename, index=False)
    
    print(f"Success! Merged {len(file_list)} files into '{output_filename}'.")
else:
    print("No files matching 'pinnacle_*.csv' were found.")
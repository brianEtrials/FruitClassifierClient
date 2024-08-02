import os
import cv2
import numpy as np
import pandas as pd
from skimage.measure import label, regionprops

# Directory containing images
# image_dir = r'C:\Users\brian\Downloads\NavelOrangesJPG\NavelOrangesJPG'
# image_dir = r'C:\Users\brian\Downloads\MandarinsJPG\MandarinsJPG\mandarin_subset'
# image_dir = r'C:\Users\brian\Downloads\grannysmith\grannysmith\granny_smith_subset'
# image_dir = r'C:\Users\brian\Downloads\honeycrisp\honeycrisp\honeycrisp_subset'
# image_dir = r'C:\Users\brian\OneDrive\Desktop\cherry tomatos\cherry_tomato_subset'
# image_dir = r'C:\Users\brian\OneDrive\Desktop\roma tomatos'
image_dir = r'C:\Users\brian\Downloads\Fuji\JPEG'
# Function to process and extract features from an image
def process_image(image_path):
    # Load the image
    image = cv2.imread(image_path)

    ############################################################################################
    # Measurements
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Otsu's thresholding
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Label the connected components
    label_image = label(binary)

    # Extract region properties
    regions = regionprops(label_image)

    # Assume the largest region is the object of interest
    region = max(regions, key=lambda r: r.area)

    # Extracting features
    area = region.area
    perimeter = region.perimeter
    major_axis_length = region.major_axis_length
    minor_axis_length = region.minor_axis_length
    eccentricity = region.eccentricity

    ###########################################################
    # Color Data
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Calculate mean and standard deviation for the entire image
    mean_rgb = image_rgb.mean(axis=(0, 1))
    std_rgb = image_rgb.std(axis=(0, 1))

    # Return the extracted features as a dictionary
    return {
        'Area': area,
        'Perimeter': perimeter,
        'Major Axis Length': major_axis_length,
        'Minor Axis Length': minor_axis_length,
        'Eccentricity': eccentricity,
        'ROI Mean (R)': mean_rgb[0],
        'ROI Mean (G)': mean_rgb[1],
        'ROI Mean (B)': mean_rgb[2],
        'ROI Std Dev (R)': std_rgb[0],
        'ROI Std Dev (G)': std_rgb[1],
        'ROI Std Dev (B)': std_rgb[2],
    }

# List to store the results
results = []

# Iterate through all files in the directory
for filename in os.listdir(image_dir):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
        image_path = os.path.join(image_dir, filename)
        features = process_image(image_path)
        results.append(features)

# Create a DataFrame from the results
df_results = pd.DataFrame(results)

# Adjust display options
pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.max_rows', None)     # Show all rows

# Display the DataFrame
print(df_results)

# Optionally, save the DataFrame to a CSV file
output_csv_path = r'C:\Users\brian\Downloads\NavelOrangesJPG\NavelOrangesJPG\fuji.csv'
df_results.to_csv(output_csv_path, index=False)

print(f'Data saved to {output_csv_path}')

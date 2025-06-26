import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os


# Add the core directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))


from util import get_path

# Load the image (ensure it's in the same directory or give full path)
img = cv2.imread(get_path(r'images_large_es1c07875_0002.jpeg'))
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Thresholding to extract black dots (data points)
_, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)

# Find contours (small blobs corresponding to scatter points)
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Create a dataframe for storing data points
data = []

# Dimensions of the full image
h, w = gray.shape

# Number of rows and columns in the plot grid
n_rows = 28  # commodities
n_cols = 6   # variables

# Calculate width and height of each cell
cell_h = h // n_rows
cell_w = w // n_cols

# Commodity and variable names (must match the image layout manually)
commodities = [
    'Aluminum', 'Chromium', 'Cobalt', 'Copper', 'Gallium', 'Gold', 'Iridium',
    'Iron', 'Lithium', 'Magnesium', 'Molybdenum', 'Nickel', 'Palladium',
    'Platinum', 'Rhodium', 'Ruthenium', 'Silicon', 'Silver', 'Tantalum',
    'Tin', 'Titanium', 'Tungsten', 'Vanadium', 'Zinc', 'Zirconium'
]

# Add missing rows manually to match 28 (2 rows are blank/combined)
while len(commodities) < n_rows:
    commodities.insert(-1, "Unknown")

variables = [
    'Log ore mined', 'Waste to ore ratio', 'Log ore grade',
    'Concentrator recovery rate', 'Refinery recovery rate', 'Unknown'
]

# Loop over all contours
for cnt in contours:
    x, y, w_box, h_box = cv2.boundingRect(cnt)
    cx = x + w_box // 2
    cy = y + h_box // 2

    # Get grid position
    row = cy // cell_h
    col = cx // cell_w

    if row < len(commodities) and col < len(variables):
        data.append({
            'Commodity': commodities[row],
            'Variable': variables[col],
            'Pixel_X': cx,
            'Pixel_Y': cy
        })

# Convert to DataFrame and save as CSV
df = pd.DataFrame(data)
df.to_csv('extracted_datapoints.csv', index=False)
print("Saved extracted datapoints to 'extracted_datapoints.csv'")
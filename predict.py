import cv2 as cv
import numpy as np
import glob
import os


# Input folder containing your 74 board images
input_folder = "boards"

# Define function to split the board into 5x5 tiles
def get_tiles(image):
    tiles = []
    
    # Resize to ensure board fits 500x500 pixels (adjustable if needed)
    board_resized = cv.resize(image, (500, 500))
    tile_size = board_resized.shape[0] // 5  # Each tile is 100x100 if 500x500 board
    
    for y in range(5):
        row = []
        for x in range(5):
            margin = 5  # Small margin to avoid edges
            y_start = max(y * tile_size + margin, 0)
            y_end = min((y + 1) * tile_size - margin, board_resized.shape[0])
            x_start = max(x * tile_size + margin, 0)
            x_end = min((x + 1) * tile_size - margin, board_resized.shape[1])
            tile = board_resized[y_start:y_end, x_start:x_end]
            row.append(tile)
        tiles.append(row)
    return tiles

# Your provided terrain classification logic using median HSV
def get_terrain(tile):
    tile_blurred = cv.GaussianBlur(tile, (5, 5), 0)
    hsv_tile = cv.cvtColor(tile_blurred, cv.COLOR_BGR2HSV)
    hue, saturation, value = np.median(hsv_tile, axis=(0, 1))

    # Thresholds you provided
    if 21.5 < hue < 27.5 and 225 < saturation < 255.5 and 104 < value < 210:
        return "fi"
    if 25 < hue < 60 and 88.5 < saturation < 247 and 24.5 < value < 78.5:
        return "fo"
    if 43.5 < hue < 120 and 221.5 < saturation < 275 and 115 < value < 204.5:
        return "la"
    if 34.5 < hue < 46.5 and 150 < saturation < 260 and 91.5 < value < 180:
        return "gr"
    if 16.5 < hue < 27 and 66 < saturation < 180 and 75 < value < 138.5:
        return "sw"
    if 19.5 < hue < 27 and 39.5 < saturation < 150 and 29.5 < value < 80:
        return "mi"
    if 17.5 < hue < 35 and 110.5 < saturation < 225 and 100 < value < 184.5:
        return "ta"
    if 19.5 < hue < 39.5 and 40.5 < saturation < 140.5 and 44.5 < value < 182.5:
        return "ho"
    
    return "unknown"






# Process all images in the folder
image_files = glob.glob(os.path.join(input_folder, "*.jpg"))  # Fix: Search for .jpg files in the folder
print(f"Found {len(image_files)} images.")

for image_path in image_files:
    image = cv.imread(image_path)
    if image is None:
        print(f"Error reading {image_path}")
        continue

    print(f"\nProcessing {os.path.basename(image_path)}:")

    tiles = get_tiles(image)
    
    for y, row in enumerate(tiles):
        for x, tile in enumerate(row):
            terrain = get_terrain(tile)
            print(f"Tile ({x}, {y}): {terrain}")

print("Done with all images.")
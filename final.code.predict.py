import cv2 as cv
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def find_crowns(game_img, templates_path):
    all_rectangles = []
    for template_path in templates_path:
        crown_template = cv.imread(template_path, cv.IMREAD_UNCHANGED)
        result = cv.matchTemplate(game_img, crown_template, cv.TM_CCOEFF_NORMED)
        threshold = 0.57
        yloc, xloc = np.where(result >= threshold)

        for i in range(len(xloc)):
            x, y = xloc[i], yloc[i]
            w = crown_template.shape[1]
            h = crown_template.shape[0]
            all_rectangles.append([int(x), int(y), int(w), int(h)])

    rectangles = cv.groupRectangles(np.array(all_rectangles), 1, 0.2)[0]
    return rectangles

def train_terrain_classifier(data_path):
    df = pd.read_csv(data_path)
    X = df[['h_median', 's_median', 'v_median', 'h_mean', 's_mean', 'v_mean']]
    y = df['terrain_type']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    return clf

def extract_tile_features(tile):
    hsv = cv.cvtColor(tile, cv.COLOR_BGR2HSV)
    h_median = np.median(hsv[:, :, 0])
    s_median = np.median(hsv[:, :, 1])
    v_median = np.median(hsv[:, :, 2])
    h_mean = np.mean(hsv[:, :, 0])
    s_mean = np.mean(hsv[:, :, 1])
    v_mean = np.mean(hsv[:, :, 2])
    return [h_median, s_median, v_median, h_mean, s_mean, v_mean]

def analyze_board(board_num, templates_path, data_path):
    image_path = f"C:/Users/salah/OneDrive - Aalborg Universitet/Desktop/King Domino dataset/Cropped and perspective corrected boards/{board_num}.jpg"
    game_img = cv.imread(image_path, cv.IMREAD_UNCHANGED)
    
    if game_img is None:
        print(f"Could not load image for board number {board_num}")
        return

    # Detect crowns
    rectangles = find_crowns(game_img, templates_path)

    # Draw rectangles around detected crowns
    for (x, y, w, h) in rectangles:
        cv.rectangle(game_img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the image with detected crowns
    cv.imshow(f'Detected Crowns on Board {board_num}', game_img)
    cv.waitKey(0)
    cv.destroyAllWindows()

    # Calculate score
    rows, cols = 5, 5
    tile_height = game_img.shape[0] // rows
    tile_width = game_img.shape[1] // cols

    # Load terrain classifier
    clf = train_terrain_classifier(data_path)

    # Extract tile features and predict terrain types
    tiles = []
    for y in range(rows):
        for x in range(cols):
            x_start = x * tile_width
            y_start = y * tile_height
            x_end = (x + 1) * tile_width
            y_end = (y + 1) * tile_height
            tile = game_img[y_start:y_end, x_start:x_end]
            tiles.append(tile)

    features = []
    for tile in tiles:
        features.append(extract_tile_features(tile))
    features_df = pd.DataFrame(features, columns=['h_median', 's_median', 'v_median', 'h_mean', 's_mean', 'v_mean'])
    predicted_terrain = clf.predict(features_df)

    # Create grid with terrain types and crown counts
    grid = []
    for _ in range(rows):
        grid.append([{'Type': None, 'CrownCount': 0} for _ in range(cols)])

    for idx, (y, x) in enumerate([(y, x) for y in range(rows) for x in range(cols)]):
        grid[y][x]['Type'] = predicted_terrain[idx]

    # Count crowns in each tile
    for y in range(rows):
        for x in range(cols):
            x_start = x * tile_width
            y_start = y * tile_height
            x_end = (x + 1) * tile_width
            y_end = (y + 1) * tile_height
            tile_rect = (x_start, y_start, tile_width, tile_height)
            crown_count = 0
            for (cx, cy, cw, ch) in rectangles:
                if (cx >= x_start and cy >= y_start and cx + cw <= x_end and cy + ch <= y_end):
                    crown_count += 1
            grid[y][x]['CrownCount'] = crown_count

    # Calculate score using flood fill algorithm
    visited = [[False for _ in range(cols)] for _ in range(rows)]
    total_score = 0

    def flood_fill(y, x, terrain_type):
        if y < 0 or y >= rows or x < 0 or x >= cols or visited[y][x] or grid[y][x]['Type'] != terrain_type:
            return 0, 0
        visited[y][x] = True
        area = 1
        crowns = grid[y][x]['CrownCount']
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for dy, dx in directions:
            a, c = flood_fill(y + dy, x + dx, terrain_type)
            area += a
            crowns += c
        return area, crowns

    for y in range(rows):
        for x in range(cols):
            if not visited[y][x] and grid[y][x]['CrownCount'] > 0:
                area, crowns = flood_fill(y, x, grid[y][x]['Type'])
                total_score += area * crowns

    print(f"Total score for board {board_num}: {total_score}")

def main():
    templates_path = [
        "C:/Users/salah/OneDrive - Aalborg Universitet/Desktop/crown_template/Crown0.jpg",
        "C:/Users/salah/OneDrive - Aalborg Universitet/Desktop/crown_template/Crown1.jpg",
        "C:/Users/salah/OneDrive - Aalborg Universitet/Desktop/crown_template/Crown2.jpg",
        "C:/Users/salah/OneDrive - Aalborg Universitet/Desktop/crown_template/Crown3.jpg"
    ]
    data_path = "C:/Users/salah/OneDrive - Aalborg Universitet/Desktop/tile_features.csv"

    board_num = int(input("Enter the board number to analyze: "))
    analyze_board(board_num, templates_path, data_path)

if __name__ == "__main__":
    main()
import cv2 as cv
import numpy as np
import os

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

def analyze_board(board_num, templates_path):
    # Construct the image path using the board number
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

def main():
    # List of paths to your crown templates
    templates_path = [
        "C:/Users/salah/OneDrive - Aalborg Universitet/Desktop/crown_template/Crown0.jpg",
        "C:/Users/salah/OneDrive - Aalborg Universitet/Desktop/crown_template/Crown1.jpg",
        "C:/Users/salah/OneDrive - Aalborg Universitet/Desktop/crown_template/Crown2.jpg",
        "C:/Users/salah/OneDrive - Aalborg Universitet/Desktop/crown_template/Crown3.jpg"
    ]

    # Get user input for the board number
    board_num = int(input("Enter the board number to analyze: "))

    # Process the specified board
    analyze_board(board_num, templates_path)

if __name__ == "__main__":
    main()
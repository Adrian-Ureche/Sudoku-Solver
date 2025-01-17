import cv2
import numpy as np
from scipy import ndimage
import math
from includes import sudokuSolver
import copy
from keras.models import load_model

model = load_model('includes/numbersDetection.h5')


# Returns the largest contour because we assume the Sudoku board is the largest contour.
def conturul_maxim(conturLst):
    areaMax = 0     # Initialize the maximum area determined by the contour as 0 (to be overwritten later)
    for contur in conturLst:    # Iterate through the list of contours
        area = cv2.contourArea(contur)   # Obtain the area determined by the contour
        if area > areaMax:  # If the determined area is larger than the maximum found so far
            areaMax = area  # Update the maximum area
            conturMax = contur  # Save the pair in conturMax
    print(conturMax)
    return conturMax


# Returns the 4 corners of a contour. These 4 corners will represent the corners of the Sudoku board.
def colturile_conturului(conturMax):
    max_iter = 200
    coeficient = 1
    while max_iter > 0 and coeficient >= 0:
        max_iter -= 1
        epsilon = coeficient * cv2.arcLength(conturMax, True)
        poly_approx = cv2.approxPolyDP(conturMax, epsilon, True)
        hull = cv2.convexHull(poly_approx)
        if len(hull) == 4:
            return hull
        else:
            if len(hull) > 4:
                coeficient += .01
            else:
                coeficient -= .01
    return None


# This function is used to make predicting a number easier.
# The Sudoku board will be split into 9*9 images, and each will pass through this function before prediction.
def claritate_numere(image):
    image = image.astype(np.uint8)
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=8)
    sizes = stats[:, -1]
    if len(sizes) <= 1:
        blank_image = np.zeros(image.shape)
        blank_image.fill(255)
        return blank_image
    max_label = 1
    max_size = sizes[1]
    for i in range(2, nb_components):
        if sizes[i] > max_size:
            max_label = i
            max_size = sizes[i]
    output_image = np.zeros(output.shape)
    output_image.fill(255)
    output_image[output == max_label] = 0
    return output_image


# Find the coordinates and position of each corner.
def reg_colturi(colturi):
    board = np.zeros((4, 2), np.float32)    # Define the board as a 4*2 array with all values set to 0
    colturi = colturi.reshape(4, 2) # Reshape colturi to a 4*2 array, keeping the order of values

    # Find the coordinates of the top-left corner
    sum = 10000
    index = 0
    for i in range(4):
        if colturi[i][0] + colturi[i][1] < sum:
            sum = colturi[i][0] + colturi[i][1]
            index = i
    board[0] = colturi[index]
    colturi = np.delete(colturi, index, 0)

    # Find the coordinates of the bottom-right corner
    sum = 0
    for i in range(3):
        if colturi[i][0] + colturi[i][1] > sum:
            sum = colturi[i][0] + colturi[i][1]
            index = i
    board[2] = colturi[index]
    colturi = np.delete(colturi, index, 0)

    # Two corners remain. The top-right corner has the larger X-coordinate.
    if colturi[0][0] > colturi[1][0]:
        board[1] = colturi[0]   # top-right
        board[3] = colturi[1]   # bottom-left
    else:
        board[1] = colturi[1]   # top-right
        board[3] = colturi[0]   # bottom-left

    board = board.reshape(4, 2)

    return board


# Predict numbers row by row by cropping each number into a separate image.
def predictie(crop_image):
    mat = []    # Empty matrix to store numbers from the game, place 0 where the cell is empty
    for i in range(9):
        row = []
        for j in range(9):
            row.append(0)
        mat.append(row) # Fill the 9*9 matrix with zeros

    height = crop_image.shape[0] // 9   # Height of a single cell
    width = crop_image.shape[1] // 9    # Width of a single cell

    # Iterate through the 81 cells of the Sudoku board
    for i in range(9):
        for j in range(9):
            # Crop the cell for prediction
            casuta = crop_image[height * i : height * (i + 1) , width * j : width * (j + 1)]

            # Remove the lines around the cell that form the Sudoku grid

            # Top
            while np.sum(casuta[0]) <= 0.4 * casuta.shape[1] * 255:
                casuta = casuta[1:]
            # Bottom
            while np.sum(casuta[:, -1]) <= 0.4 * casuta.shape[1] * 255:
                casuta = np.delete(casuta, -1, 1)
            # Left
            while np.sum(casuta[:, 0]) <= 0.4 * casuta.shape[0] * 255:
                casuta = np.delete(casuta, 0, 1)
            # Right
            while np.sum(casuta[-1]) <= 0.4 * casuta.shape[0] * 255:
                casuta = casuta[:-1]

            # Prepare the image for prediction
            casuta = cv2.bitwise_not(casuta)
            casuta = claritate_numere(casuta)

            # Resize to 28*28
            casuta = cv2.resize(casuta, (28, 28))

            center_width = casuta.shape[1] // 2     # Center of cell height
            center_height = casuta.shape[0] // 2    # Center of cell width
            # Coordinates of the rectangle where the digit should be
            x_start = center_height // 2
            x_end = center_height // 2 + center_height
            y_start = center_width // 2
            y_end = center_width // 2 + center_width
            # Image that should contain the digit, used to check if the cell is empty
            center_region = casuta[x_start:x_end, y_start:y_end]

            # Place 0 where cells are empty
            if center_region.sum() >= center_width * center_height * 255 - 255:
                mat[i][j] = 0
                continue

            # Prepare for prediction
            _, casuta = cv2.threshold(casuta, 200, 255, cv2.THRESH_BINARY)
            casuta = casuta.astype(np.uint8)

            casuta = casuta.reshape(-1, 28, 28, 1)
            casuta = casuta.astype(np.float32)
            casuta /= 255

            # Make the prediction. The prediction is a list containing the probability that the input is one of the digits 1, 2, 3 ...
            predictie = model.predict([casuta])
            # The index of the element in the list with the highest value is the most likely digit in the cell.
            # In the list, position 0 is the probability of the number being 1, position 1 is the probability of the number being 2, etc. So, we add +1.
            mat[i][j] = np.argmax(predictie) + 1
    return mat


def scrie_solutie_pe_imagine(image, mat_rez, mat_init):
    width = image.shape[1] // 9     # Width of a single cell
    height = image.shape[0] // 9    # Height of a single cell

    # Iterate through the 81 cells of the Sudoku board
    for i in range(9):
        for j in range(9):
            # If the value in the initial matrix is different from 0, it means this cell already has a number, so we skip it
            if mat_init[i][j] != 0:
                continue

            text = str(mat_rez[i][j])   # Convert the digit to be written on the image to a string
            poz_x = width // 15     # X-coordinate within the cell where the text will be written
            poz_y = height // 15    # Y-coordinate within the cell where the text will be written
            # Determine the dimensions of the bounding box that will contain the text
            (text_height, text_width), baseLine = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
            # Adjust the font size based on the board's size
            ajustare_font = 0.5 * min(width, height) / max(text_height, text_width)
            text_height *= ajustare_font
            text_width *= ajustare_font
            # X-coordinate on the image where the text will be written
            coordonata_x = width*j + math.floor((width - text_width) / 2) + poz_x
            # Y-coordinate on the image where the text will be written
            coordonata_y = height*(i+1) - math.floor((height - text_height) / 2) + poz_y
            # Write the text on the image at the calculated coordinates with the calculated font size and green color
            image = cv2.putText(image, text, (coordonata_x, coordonata_y), cv2.FONT_HERSHEY_SIMPLEX, ajustare_font, (0, 255, 0), 2)

    return image


# Here you can replace the path to the image. You can use i2 or i3 for other tests; the images are in the "images" folder
sudoku = cv2.imread("images/i2.jpg") # Read the image
image_gray = cv2.cvtColor(sudoku, cv2.COLOR_BGR2GRAY)   # Convert the image to grayscale
image_blur = cv2.GaussianBlur(image_gray, (5, 5), 2)    # Add a blur effect for easier detection
image_threshold = cv2.adaptiveThreshold(image_blur, 255, 1, 1, 11, 2)   # Apply adaptiveThreshold to later detect contours

conturLst, _ = cv2.findContours(image_threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)    # Find all contours and save them in a list of coordinates
conturMax = conturul_maxim(conturLst)   # Find the largest contour, assuming it represents the Sudoku board
colturi = colturile_conturului(conturMax)   # Extract the corners of the contour

board = reg_colturi(colturi) # Store the coordinates of each corner in a list: [0]->top-left, [1]->bottom-left, [2]->bottom-right, [3]->top-right
    
(ss, sj, dj, ds) = board    # Calculate the length of each side of the Sudoku board
width_A = np.sqrt(((dj[0] - ds[0]) ** 2) + ((dj[1] - ds[1]) ** 2))
width_B = np.sqrt(((sj[0] - ss[0]) ** 2) + ((sj[1] - ss[1]) ** 2))
height_A = np.sqrt(((sj[0] - dj[0]) ** 2) + ((sj[1] - dj[1]) ** 2))
height_B = np.sqrt(((ss[0] - ds[0]) ** 2) + ((ss[1] - ds[1]) ** 2))

max_width = max(int(width_A), int(width_B))     # Determine the maximum dimensions
max_height = max(int(height_A), int(height_B))

# Build a list of coordinates that will be used to extract the Sudoku board from the image
sudokuCoord = np.array([[0, 0], [max_width - 1, 0], [max_width - 1, max_height - 1], [0, max_height - 1]], np.float32)

cv2.drawContours(sudoku, colturi, -1, (0, 255, 0), 15)  # Draw 4 green circles in each corner of the Sudoku board
M = cv2.getPerspectiveTransform(board, sudokuCoord)     # Build the transformation matrix
crop_image = cv2.warpPerspective(sudoku, M, (max_width, max_height))    # Extract the Sudoku board and reshape it as a rectangle
crop_image_wrap = np.copy(crop_image)   # Make a copy of the board to write the solution on

# Apply the following effects to the image to facilitate digit recognition
crop_image = cv2.cvtColor(crop_image, cv2.COLOR_BGR2GRAY)
crop_image = cv2.GaussianBlur(crop_image, (5, 5), 2)
crop_image = cv2.adaptiveThreshold(crop_image, 255, 1, 1, 11, 2)
crop_image = cv2.bitwise_not(crop_image)
_, crop_image = cv2.threshold(crop_image, 10, 255, cv2.THRESH_BINARY)

mat_rez = predictie(crop_image) # Extract the numbers from the image and store them in a matrix; use 0 for empty cells
mat_nerez = copy.deepcopy(mat_rez)  # Make a copy of the matrix to track changes compared to the initial one

sudokuSolver.solve_sudoku(mat_rez)  # Solve the Sudoku puzzle, updating the matrix with the solution

if sudokuSolver.all_board_non_zero(mat_rez):    # Check if the Sudoku puzzle was solved
    crop_image_wrap = scrie_solutie_pe_imagine(crop_image_wrap, mat_rez, mat_nerez) # Write the solution on the cropped Sudoku board
    
# Apply the inverse transformation. Place the completed Sudoku board back onto the original image
result_sudoku = cv2.warpPerspective(crop_image_wrap, M, (sudoku.shape[1], sudoku.shape[0]), flags=cv2.WARP_INVERSE_MAP)
sudoku_rezolvat = np.where(result_sudoku.sum(axis=-1, keepdims=True) != 0, result_sudoku, sudoku)

cv2.imwrite('Sudoku_Rezolvat.png', sudoku_rezolvat)     # Finally, save the image!

cv2.imshow('Sudoku_Rezolvat', sudoku_rezolvat)
cv2.waitKey(0)
cv2.destroyAllWindows()

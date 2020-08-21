from cv2 import cv2
import imutils
import numpy as np
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Param
max_size = 5000
min_size = 900

# Load image
img = cv2.imread('test2.jpg', cv2.IMREAD_COLOR)

# Resize image
#img = cv2.resize(img, (700, 580)) #Done with img = test5
img = cv2.resize(img, (620, 480)) #Done with img = test2

# Edge detection
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # convert to grey scale
gray = cv2.bilateralFilter(gray, 11, 17, 17)  # Blur to reduce noise
edged = cv2.Canny(gray, 30, 200)  # Perform Edge detection

# find contours in the edged image, keep only the largest
# ones, and initialize our screen contour
cnts = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
screenCnt = None

# loop over our contours
for c in cnts:
    # approximate the contour
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.05 * peri, True)

    # if our approximated contour has four points, then
    # we can assume that we have found our screen
    if len(approx) == 4 and max_size > cv2.contourArea(c) > min_size:
        screenCnt = approx
        break

if screenCnt is None:
    detected = 0
    print("No plate detected")
else:
    detected = 1

if detected == 1:
    cv2.drawContours(img, [screenCnt], -1, (0, 255, 0), 3)

    # Masking the part other than the number plate
    mask = np.zeros(gray.shape, np.uint8)
    new_image = cv2.drawContours(mask, [screenCnt], 0, 255, -1, )
    new_image = cv2.bitwise_and(img, img, mask=mask)

    # Now crop
    (x, y) = np.where(mask == 255)
    (topx, topy) = (np.min(x), np.min(y))
    (bottomx, bottomy) = (np.max(x), np.max(y))
    Cropped = gray[topx:bottomx + 1, topy:bottomy + 1]
    
    # Print number
    text = pytesseract.image_to_string(Cropped, config='--psm 11')
    print("programming_fever's License Plate Recognition\n")
    print("Detected license plate Number is:", text)
    img = cv2.resize(img, (500, 300))
    Cropped = cv2.resize(Cropped, (350, 90))
    #cv2.imshow('car', img)
    #cv2.imshow('Cropped', Cropped)
    
    # Display image
    cv2.imshow('Input image', img)
    cv2.imshow('License plate', Cropped)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

cv2.waitKey(0)

import numpy as np
import os
import cv2
from matplotlib import pyplot as plt
import argparse
import glob
from scipy.stats import mode
import argparse as ap


def imageProcess(gray, blurPara, erodeSize):
    # Contrast Streching
    hist, bins = np.histogram(gray.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_m = np.ma.masked_equal(cdf, 0)
    cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
    cdf = np.ma.filled(cdf_m, 0).astype('uint8')

    # Computing of the Contrast Streching
    result2 = cdf[gray]
    result = cv2.LUT(gray, cdf)

    # Median Blur
    blur = cv2.medianBlur(result, blurPara)

    # Binarilize with OTSU
    ret, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_OTSU)

    # Eroding by 4*4 rectrangule
    kernel = np.ones(erodeSize, np.uint8)
    erosion = cv2.erode(thresh, kernel, iterations=1)

    # Sober Edge Detection
    sobelx64f = cv2.Sobel(erosion, cv2.CV_64F, 1, 0, ksize=3)
    abs_sobel64f = np.absolute(sobelx64f)
    sobelx = np.uint8(abs_sobel64f)
    return sobelx, thresh


def waterGaugeEdgeDet(img, prodImg, threshold):
    image = img.copy()
    lines = cv2.HoughLines(prodImg, 1, np.pi / 180, threshold)
    lines1 = lines[:, 0, :]
    if lines1[1][0] >= 0:
        rotateAngle = lines1[1][1]
    else:
        rotateAngle = - lines1[1][1]
    for rho, theta in lines1[:]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv2.line(image, (x1, y1), (x2, y2), (255, 255, 255), 1)
    return lines, image, rotateAngle


def rotate_image(image, angle):
    image_size = (image.shape[1], image.shape[0])
    image_center = tuple(np.array(image_size) / 2)

    # Convert the OpenCV 3x2 rotation matrix to 3x3
    rot_mat = np.vstack(
        [cv2.getRotationMatrix2D(image_center, angle, 1.0), [0, 0, 1]]
    )

    rot_mat_notranslate = np.matrix(rot_mat[0:2, 0:2])

    # Shorthand for below calcs
    image_w2 = image_size[0] * 0.5
    image_h2 = image_size[1] * 0.5

    # Obtain the rotated coordinates of the image corners
    rotated_coords = [
        (np.array([-image_w2, image_h2]) * rot_mat_notranslate).A[0],
        (np.array([image_w2, image_h2]) * rot_mat_notranslate).A[0],
        (np.array([-image_w2, -image_h2]) * rot_mat_notranslate).A[0],
        (np.array([image_w2, -image_h2]) * rot_mat_notranslate).A[0]
    ]

    # Find the size of the new image
    x_coords = [pt[0] for pt in rotated_coords]
    x_pos = [x for x in x_coords if x > 0]
    x_neg = [x for x in x_coords if x < 0]

    y_coords = [pt[1] for pt in rotated_coords]
    y_pos = [y for y in y_coords if y > 0]
    y_neg = [y for y in y_coords if y < 0]

    right_bound = max(x_pos)
    left_bound = min(x_neg)
    top_bound = max(y_pos)
    bot_bound = min(y_neg)

    new_w = int(abs(right_bound - left_bound))
    new_h = int(abs(top_bound - bot_bound))

    # We require a translation matrix to keep the image centred
    trans_mat = np.matrix([
        [1, 0, int(new_w * 0.5 - image_w2)],
        [0, 1, int(new_h * 0.5 - image_h2)],
        [0, 0, 1]
    ])

    # Compute the tranform for the combined rotation and translation
    affine_mat = (np.matrix(trans_mat) * np.matrix(rot_mat))[0:2, :]

    # Apply the transform
    result = cv2.warpAffine(
        image,
        affine_mat,
        (new_w, new_h),
        flags=cv2.INTER_LINEAR
    )

    return result


def horizontalProjection(img):
    (h, w) = img.shape[:2]
    sumCols = []
    for j in range(h):
        col = img[j:j + 1, 0:w]  # y1:y2, x1:x2
        sumCols.append(np.sum(col))
    return sumCols


def verticalProjection(img):
    "Return a list containing the sum of the pixels in each column"
    (h, w) = img.shape[:2]
    sumCols = []
    for j in range(w):
        col = img[0:h, j:j + 1]  # y1:y2, x1:x2
        sumCols.append(np.sum(col))
    return sumCols


def horiCut(x_Proj, threshold):
    l = []
    for i in range(len(x_Proj)):
        if x_Proj[i] >= threshold * np.mean(x_Proj):
            l.append(i)
    return l[0], l[len(l) - 1]


def vertiCut(y_Proj, threshold):
    l = []
    for i in range(len(y_Proj)):
        if y_Proj[i] >= threshold * np.mean(y_Proj):
            l.append(i)
    return l[0], l[len(l) - 1]


def getLength(contourList):
    length = [0]
    for i in range(len(contourList)):
        contour = contourList[i].reshape(len(contourList[i]), 2)
        leng = contour[3][1] - contour[0][1]
        length.append(leng)
    return np.max(length), length

def scaleRead(cropImage,cropFileName):
    # Loading the image
    image = cropImage.copy()

    # Converting to Grayscale image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    procImg, binaryImg = imageProcess(gray, 5, (3, 3))

    lineStr, edge, angle = waterGaugeEdgeDet(image, procImg, 100)

    rotatedImg = rotate_image(binaryImg, angle)
    H, W = rotatedImg.shape
    # Horizontal Sobel Edge Detection
    sobelx = cv2.Sobel(rotatedImg, cv2.CV_64F, 0, 1, ksize=3)
    abs_sobel = np.absolute(sobelx)
    sobelx = np.uint8(abs_sobel)

    # Noise Filter
    blurImg = cv2.medianBlur(sobelx, 1)

    xProj = verticalProjection(blurImg)
    yProj = horizontalProjection(blurImg)

    xLeft, xRight = horiCut(xProj, 0.1)
    print(xLeft, xRight)
    ytop, ybottom = vertiCut(yProj, 0.1) 
    print(ytop, ybottom)
    height = ybottom - ytop

    cropped_image = rotatedImg[ytop:ybottom, xLeft:xRight]
    right_image = rotatedImg[ytop:ybottom,
                             int(np.mean((xLeft, xRight))):xRight]
    left_image = rotatedImg[ytop:ybottom, xLeft:int(np.mean((xLeft, xRight)))]
    left_image = cv2.flip(left_image, 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
    openLeft = cv2.morphologyEx(left_image, cv2.MORPH_OPEN, kernel)
    openRight = cv2.morphologyEx(right_image, cv2.MORPH_OPEN, kernel)

    image11, contours, hierarchy = cv2.findContours(openRight, 1, 2)

    h, l = getLength(contours)

    lengthofGauge = (height / h) * 0.05
    measureResult = "The current water level is " + str(lengthofGauge) + " m"

    rotatedImg1 = rotate_image(image, angle)
    crop = rotatedImg1[ytop:ybottom, xLeft:xRight]
    
    out_path = "./scale_detect/"
    savePath = os.path.join(out_path, cropFileName.split('/')[-1])
    plt.subplot(151), plt.imshow(image), plt.title("img_input") 
    plt.subplot(152), plt.imshow(rotatedImg,'gray'), plt.title("img_rotate")
    plt.subplot(153), plt.imshow(cropped_image,'gray'), plt.title("img_crop")
    plt.subplot(154), plt.imshow(right_image,'gray'), plt.title("img_right")
    plt.subplot(155), plt.imshow(openRight,'gray'), plt.title("right_open")
    plt.savefig(savePath,bbox_inches='tight', pad_inches=0)
    return measureResult,savePath


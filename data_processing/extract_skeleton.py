import os
import cv2
from matplotlib import pyplot as plt
import numpy as np
import math
import skimage.morphology
from skimage.morphology import skeletonize, thin, convex_hull_image, erosion, square
from skimage import img_as_ubyte
from Extractor import extract_minutiae_features
from scipy.spatial import KDTree
from PIL import Image
def _meshgrid(xgv, ygv):
    x = np.outer(np.ones_like(ygv), xgv)
    y = np.outer(ygv, np.ones_like(xgv))
    return x, y


def _type2str(type):
    depth = type & cv2.cv.CV_MAT_DEPTH_MASK
    chans = 1 + (type >> cv2.cv.CV_CN_SHIFT)

    switcher = {
        cv2.CV_8U: "8U",
        cv2.CV_8S: "8S",
        cv2.CV_16U: "16U",
        cv2.CV_16S: "16S",
        cv2.CV_32S: "32S",
        cv2.CV_32F: "32F",
        cv2.CV_64F: "64F",
        "default": "User"
    }

    r = switcher.get(depth, "User")
    r += "C" + str(chans)
    return r


def _printMatrix(m, showColumns=7):
    rows, cols = m.shape
    for i in range(int(np.ceil(cols / showColumns))):
        for y in range(rows):
            for x in range(showColumns):
                column = x + i * showColumns
                if column < cols:
      
                    print("    {0:.4f}".format(m[y, column]), end="")
            print()
        print()


def _conv(src, kernel):
    anchor = (kernel.shape[1] - kernel.shape[1] // 2 - 1, kernel.shape[0] - kernel.shape[0] // 2 - 1)
    flipped = cv2.flip(kernel, 0)
    result = cv2.filter2D(src, -1, flipped, anchor=anchor, borderType=cv2.BORDER_REPLICATE)
    return result


def MaxCurvature(_src, _mask, sigma):
    src = _src.copy()

    src = src.astype(np.float32) / 255.0

    mask = _mask

    sigma2 = np.power(sigma, 2)
    sigma4 = np.power(sigma, 4)

    # Construct filter kernels
    winsize = math.ceil(4 * sigma)
    X, Y = _meshgrid(range(-winsize, winsize), range(-winsize, winsize))

    # Construct h
    X2 = np.power(X, 2)
    Y2 = np.power(Y, 2)
    X2Y2 = X2 + Y2

    expXY = np.exp(-X2Y2 / (2 * sigma2))
    h = (1 / (2 * np.pi * sigma2)) * expXY

    # Construct hx
    Xsigma2 = -X / sigma2
    hx = np.multiply(h, Xsigma2)

    # Construct hxx
    temp = ((X2 - sigma2) / sigma4)
    hxx = np.multiply(h, temp)

    # Construct hy
    hy = hx.T

    # Construct hyy
    hyy = hxx.T

    # Construct hxy
    XY = np.multiply(X, Y)
    hxy = np.multiply(h, XY / sigma4)

    fx = -_conv(src, hx)
    fxx = _conv(src, hxx)
    fy = _conv(src, hy)
    fyy = _conv(src, hyy)
    fxy = -_conv(src, hxy)

    f1 = 0.5 * math.sqrt(2.0) * (fx + fy)
    f2 = 0.5 * math.sqrt(2.0) * (fx - fy)
    f11 = 0.5 * fxx + fxy + 0.5 * fyy
    f22 = 0.5 * fxx - fxy + 0.5 * fyy

    img_h = src.shape[0]
    img_w = src.shape[1]

    k1 = np.zeros(src.shape, dtype=np.float32)
    k2 = np.zeros(src.shape, dtype=np.float32)
    k3 = np.zeros(src.shape, dtype=np.float32)
    k4 = np.zeros(src.shape, dtype=np.float32)

    # Iterate over the image
    for x in range(img_w):
        for y in range(img_h):
            p = (y, x)
            if mask[p] > 0:
                k1[p] = fxx[p] / np.power(1 + np.power(fx[p], 2), 1.5)
                k2[p] = fyy[p] / np.power(1 + np.power(fy[p], 2), 1.5)
                k3[p] = f11[p] / np.power(1 + np.power(f1[p], 2), 1.5)
                k4[p] = f22[p] / np.power(1 + np.power(f2[p], 2), 1.5)

    # Scores
    Wr = 0
    Vt = np.zeros(src.shape, dtype=np.float32)
    pos_end = 0

    # Continue converting the rest of the C++ code as above...
    # Horizontal direction
    for y in range(img_h):
        Wr = 0
        for x in range(img_w):
            p = (y, x)
            bla = k1[p] > 0

            if bla:
                Wr += 1

            if Wr > 0 and (x == img_w - 1 or not bla):
                pos_end = x if x == img_w - 1 else x - 1
                pos_start = pos_end - Wr + 1  # Start pos of concave

                pos_max = 0
                max_val = float('-inf')
                for i in range(pos_start, pos_end + 1):
                    value = k1[(y, i)]
                    if value > max_val:
                        pos_max = i
                        max_val = value

                Scr = k1[(y, pos_max)] * Wr
                Vt[(y, pos_max)] += Scr
                Wr = 0

    # Vertical direction
    for x in range(img_w):
        Wr = 0
        for y in range(img_h):
            p = (y, x)
            bla = k2[p] > 0

            if bla:
                Wr += 1

            if Wr > 0 and (y == img_h - 1 or not bla):
                pos_end = y if y == img_h - 1 else y - 1
                pos_start = pos_end - Wr + 1  # Start pos of concave

                pos_max = 0
                max_val = float('-inf')
                for i in range(pos_start, pos_end + 1):
                    value = k2[(i, x)]
                    if value > max_val:
                        pos_max = i
                        max_val = value

                Scr = k2[(pos_max, x)] * Wr
                Vt[(pos_max, x)] += Scr
                Wr = 0

    pos_x_end = 0
    pos_y_end = 0

    # Direction \ .
    for start in range(img_h + img_w - 1):
        # Initial values
        if start < img_w:
            x = start
            y = 0
        else:
            x = 0
            y = start - img_w + 1

        done = False
        Wr = 0

        while not done:
            p = (y, x)
            bla = k3[p] > 0
            if bla:
                Wr += 1

            if Wr > 0 and (y == img_h - 1 or x == img_w - 1 or not bla):
                if y == img_h - 1 or x == img_w - 1:
                    # Reached edge of image
                    pos_x_end = x
                    pos_y_end = y
                else:
                    pos_x_end = x - 1
                    pos_y_end = y - 1

                pos_x_start = pos_x_end - Wr + 1
                pos_y_start = pos_y_end - Wr + 1

                rect = np.s_[pos_y_start:pos_y_end + 1, pos_x_start:pos_x_end + 1]
                dd = k3[rect]
                d = np.diagonal(dd)

                max_val = float('-inf')
                pos_max = 0
                for i in range(len(d)):
                    value = d[i]
                    if value > max_val:
                        pos_max = i
                        max_val = value

                pos_x_max = pos_x_start + pos_max
                pos_y_max = pos_y_start + pos_max
                Scr = k3[(pos_y_max, pos_x_max)] * Wr

                Vt[(pos_y_max, pos_x_max)] += Scr
                Wr = 0

            if x == img_w - 1 or y == img_h - 1:
                done = True
            else:
                x += 1
                y += 1

    # Direction /
    for start in range(img_h + img_w - 1):
        # Initial values
        if start < img_w:
            x = start
            y = img_h - 1
        else:
            x = 0
            y = img_w + img_h - start - 2

        done = False
        Wr = 0

        while not done:
            p = (y, x)
            bla = k4[p] > 0
            if bla:
                Wr += 1

            if Wr > 0 and (y == 0 or x == img_w - 1 or not bla):
                if y == 0 or x == img_w - 1:
                    # Reached edge of image
                    pos_x_end = x
                    pos_y_end = y
                else:
                    pos_x_end = x - 1
                    pos_y_end = y + 1

                pos_x_start = pos_x_end - Wr + 1
                pos_y_start = pos_y_end + Wr - 1

                rect = np.s_[pos_y_end:pos_y_start + 1, pos_x_start:pos_x_end + 1]
                roi = k4[rect]
                dd = np.flip(roi, 0)
                d = np.diagonal(dd)

                max_val = float('-inf')
                pos_max = 0
                for i in range(len(d)):
                    value = d[i]
                    if value > max_val:
                        pos_max = i
                        max_val = value

                pos_x_max = pos_x_start + pos_max
                pos_y_max = pos_y_start - pos_max

                Scr = k4[(pos_y_max, pos_x_max)] * Wr

                if pos_y_max < 0:
                    pos_y_max = 0

                Vt[(pos_y_max, pos_x_max)] += Scr
                Wr = 0

            if x == img_w - 1 or y == 0:
                done = True
            else:
                x += 1
                y -= 1

    Cd1 = np.zeros(src.shape, np.float32)
    Cd2 = np.zeros(src.shape, np.float32)
    Cd3 = np.zeros(src.shape, np.float32)
    Cd4 = np.zeros(src.shape, np.float32)
    for x in range(2, src.shape[1] - 3):
        for y in range(2, src.shape[0] - 3):
            p = (y, x)
            Cd1[p] = min(max(Vt[(y, x + 1)], Vt[(y, x + 2)]), max(Vt[(y, x - 1)], Vt[(y, x - 2)]))
            Cd2[p] = min(max(Vt[(y + 1, x)], Vt[(y + 2, x)]), max(Vt[(y - 1, x)], Vt[(y - 2, x)]))
            Cd3[p] = min(max(Vt[(y - 1, x - 1)], Vt[(y - 2, x - 2)]), max(Vt[(y + 1, x + 1)], Vt[(y + 2, x + 2)]))
            Cd4[p] = min(max(Vt[(y - 1, x + 1)], Vt[(y - 2, x + 2)]), max(Vt[(y + 1, x - 1)], Vt[(y + 2, x - 2)]))

    # Connection of vein centres

    veins = np.zeros(src.shape, np.float32)
    for x in range(src.shape[1]):
        for y in range(src.shape[0] - 3):
            p = (y, x)
            veins[p] = max(max(Cd1[p], Cd2[p]), max(Cd3[p], Cd4[p]))

    dst = veins.copy()

    return dst


# def image_enhaced():
#     image_enhancer = ImageEnhancer()
#     img = cv2.imread('1_line.jpg', cv2.COLOR_BGR2GRAY)
#     out = image_enhancer.enhance(img, resize=False)     # run image enhancer
#     image_enhancer.save_enhanced_image('1_enhanced.jpg')   # save output
#     # FeaturesTerminations, FeaturesBifurcations = fingerprint_feature_extractor.extract_minutiae_features(img, spuriousMinutiaeThresh=10, invertImage=False, showResult=True, saveResult=True)


def feature_extractor(img, isEndpoint, isBipoint, path):
    # img = cv2.cvtColor(img, cv2.IMREAD_GRAYSCALE)
    DispImg, result, FeaturesTerminations, FeaturesBifurcations, FeaturesCross = extract_minutiae_features(img,
                                                                                                           isEndpoint,
                                                                                                           isBipoint,
                                                                                                           path,
                                                                                                           spuriousMinutiaeThresh=10,
                                                                                                           invertImage=False,
                                                                                                           showResult=True,
                                                                                                           saveResult=True
                                                                                                           )
    minutiae = []
    if isEndpoint:
        minutiae += FeaturesTerminations
    if isBipoint:
        minutiae += (FeaturesBifurcations + FeaturesCross)

    return DispImg, result, minutiae



def draw_curve_vein(finger):
    # finger = cv2.imread('/media/nguyendung/NguyenDung/FingerVein/VERA/VERA-fingervein/cropped/bf/001-M/001_R_2.png', cv2.IMREAD_GRAYSCALE)
    mask = np.ones(finger.shape, np.uint8)  # Locus space
    result = MaxCurvature(finger, mask, 8)

    _, max_val = cv2.minMaxLoc(result)[:2]
    print(max_val)
    result = ((result * 255.0 / max_val) * 20).astype(np.uint8)

    cv2.imwrite("R2_draw_curve_vein.png", result)
    return result


def skeleton(img):
    _, binary_img = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)
    binary_img = binary_img / 255.0
    skeleton = skeletonize(binary_img)
    # Convert back to 8-bit image (0 and 255)
    skeleton = img_as_ubyte(skeleton)
    return skeleton


def smooth_image(skeleton_image):
    kernel_size = 3
    structuring_element = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    # Perform dilation
    dilated_image = cv2.dilate(skeleton_image, structuring_element, iterations=4)
    # Perform erosion
    smoothed_image = cv2.erode(dilated_image, structuring_element, iterations=3)
    # Save the smoothed image
    # cv2.imwrite('smoothed_image.jpg', smoothed_image)
    return smoothed_image

def extract_minutiae(image, isEndpoint, isBipoint, path):
    print(path)
    # file = os.listdir(path)
    # print(file)
    file = path[:-4]
    print(file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    curve = draw_curve_vein(image)
    cv2.imwrite(file+'.png', curve)



if __name__ == "__main__":
    targetdir = r'C:\Users\CVlab\Documents\01_2023\01_2023_BiosyntheticData\finger_vein_Visualization\Finger_vein_visualization\diffusion_vera'
    files = os.listdir(targetdir)
    for file in files:
        path = targetdir + file
        img1 = cv2.imread('diffusion_vera/'+file)
        #img1 = cv2.imread('M570801-FV3LNO1_ori.bmp')
        #print(img1.shape)# 0 for grayscale
        extract_minutiae(img1, False, True, file)
        #cv2.imshow('test', smooth1)
        #cv2.waitKey(0)
        #cv2.imwrite('ridge pattern_test1.png', smooth1)

        # img2 = cv2.imread('M820512-FV3LNO1_ori.bmp')  # 0 for grayscale
        # _, _, smooth2, minutiae2 = extract_minutiae(img2, False, True)
        # cv2.imshow('test2', smooth2)
        # cv2.waitKey(0)
        #cv2.imwrite('pa.jpg', smooth2)

        #BFMatcher_matchingScore_UI(smooth1, minutiae1, smooth2, minutiae2, True)






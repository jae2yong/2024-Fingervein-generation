import cv2
import numpy as np
import skimage.morphology
from skimage.morphology import convex_hull_image, erosion
from skimage.morphology import square
import math

class MinutiaeFeature(object):
    def __init__(self, locX, locY, Orientation, Type):
        self.locX = locX
        self.locY = locY
        self.Orientation = Orientation
        self.Type = Type

class FingerprintFeatureExtractor(object):
    def __init__(self):
        self._mask = []
        self._skel = []
        self.minutiaeTerm = []
        self.minutiaeBif = []
        self.minutiaeCross = []
        self._spuriousMinutiaeThresh = 10

    def setSpuriousMinutiaeThresh(self, spuriousMinutiaeThresh):
        self._spuriousMinutiaeThresh = spuriousMinutiaeThresh

    def __skeletonize(self, img):
        img = np.uint8(img > 128)
        skel = skimage.morphology.skeletonize(img)
        skel = np.uint8(skel) * 255

        kernel_size = 3
        structuring_element = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))

        # Perform dilation
        dilated_image = cv2.dilate(skel, structuring_element, iterations=5)

        # Perform erosion
        smoothed_image = cv2.erode(dilated_image, structuring_element, iterations=2)
        
        smoothed_image = np.uint8(smoothed_image > 200)
        self._skel = skimage.morphology.skeletonize(smoothed_image)
        self._skel = np.uint8(self._skel) * 255
        self._mask = img * 255

    def __computeAngle(self, block, minutiaeType):
        angle = []
        (blkRows, blkCols) = np.shape(block)
        CenterX, CenterY = (blkRows - 1) / 2, (blkCols - 1) / 2
        if (minutiaeType.lower() == 'termination'):
            sumVal = 0
            for i in range(blkRows):
                for j in range(blkCols):
                    if ((i == 0 or i == blkRows - 1 or j == 0 or j == blkCols - 1) and block[i][j] != 0):
                        angle.append(-math.degrees(math.atan2(i - CenterY, j - CenterX)))
                        sumVal += 1
                        if (sumVal > 1):
                            angle.append(float('nan'))
            return (angle)

        elif (minutiaeType.lower() == 'bifurcation'):
            (blkRows, blkCols) = np.shape(block)
            CenterX, CenterY = (blkRows - 1) / 2, (blkCols - 1) / 2
            angle = []
            sumVal = 0
            for i in range(blkRows):
                for j in range(blkCols):
                    if ((i == 0 or i == blkRows - 1 or j == 0 or j == blkCols - 1) and block[i][j] != 0):
                        angle.append(-math.degrees(math.atan2(i - CenterY, j - CenterX)))
                        sumVal += 1
            if (sumVal != 3):
                angle.append(float('nan'))
            return (angle)
        
        elif (minutiaeType.lower() == 'crosspoint'):
            (blkRows, blkCols) = np.shape(block)
            CenterX, CenterY = (blkRows - 1) / 2, (blkCols - 1) / 2
            angle = []
            sumVal = 0
            for i in range(blkRows):
                for j in range(blkCols):
                    if ((i == 0 or i == blkRows - 1 or j == 0 or j == blkCols - 1) and block[i][j] != 0):
                        angle.append(-math.degrees(math.atan2(i - CenterY, j - CenterX)))
                        sumVal += 1
            if (sumVal != 4):
                angle.append(float('nan'))
            return (angle) 

    def __getTerminationBifurcationCrosspoint(self):
        self._skel = self._skel == 255
        (rows, cols) = self._skel.shape
        self.minutiaeTerm = np.zeros(self._skel.shape)
        self.minutiaeBif = np.zeros(self._skel.shape)
        self.minutiaeCross = np.zeros(self._skel.shape)
        cross_kernel_1 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
        cross_kernel_2 = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]], dtype=np.uint8)

        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                if (self._skel[i][j] == 1):
                    block = self._skel[i - 1:i + 2, j - 1:j + 2]
                    block_val = np.sum(block)
                    if (block_val == 2):
                        self.minutiaeTerm[i, j] = 1
                    elif (block_val == 4):
                        self.minutiaeBif[i, j] = 1
                    elif (block_val == 5):
                        if (np.array_equal(block, cross_kernel_1) or np.array_equal(block, cross_kernel_2)):
                            self.minutiaeCross[i, j] = 1

        self._mask = convex_hull_image(self._mask > 0)
        self._mask = erosion(self._mask, square(5))  # Structuing element for mask erosion = square(5)
        self.minutiaeTerm = np.uint8(self._mask) * self.minutiaeTerm

    def __removeSpuriousMinutiae(self, minutiaeList, img):
        img = img * 0
        SpuriousMin = []
        numPoints = len(minutiaeList)
        D = np.zeros((numPoints, numPoints))
        for i in range(1,numPoints):
            for j in range(0, i):
                (X1,Y1) = minutiaeList[i]['centroid']
                (X2,Y2) = minutiaeList[j]['centroid']

                dist = np.sqrt((X2-X1)**2 + (Y2-Y1)**2)
                D[i][j] = dist
                if(dist < self._spuriousMinutiaeThresh):
                    SpuriousMin.append(i)
                    SpuriousMin.append(j)

        SpuriousMin = np.unique(SpuriousMin)
        for i in range(0,numPoints):
            if(not i in SpuriousMin):
                (X,Y) = np.int16(minutiaeList[i]['centroid'])
                img[X,Y] = 1

        img = np.uint8(img)
        return(img)

    def __cleanMinutiae(self, img):
        self.minutiaeTerm = skimage.measure.label(self.minutiaeTerm, connectivity=2)
        RP = skimage.measure.regionprops(self.minutiaeTerm)
        self.minutiaeTerm = self.__removeSpuriousMinutiae(RP, np.uint8(img))

    def __performFeatureExtraction(self):
        FeaturesTerm = []
        self.minutiaeTerm = skimage.measure.label(self.minutiaeTerm, connectivity=2)
        RP = skimage.measure.regionprops(np.uint8(self.minutiaeTerm))

        WindowSize = 2  # --> For Termination, the block size must can be 3x3, or 5x5. Hence the window selected is 1 or 2
        FeaturesTerm = []
        for num, i in enumerate(RP):
            (row, col) = np.int16(np.round(i['Centroid']))
            block = self._skel[row - WindowSize:row + WindowSize + 1, col - WindowSize:col + WindowSize + 1]
            angle = self.__computeAngle(block, 'Termination')
            if(len(angle) == 1):
                FeaturesTerm.append(MinutiaeFeature(row, col, angle, 0))

        FeaturesBif = []
        self.minutiaeBif = skimage.measure.label(self.minutiaeBif, connectivity=2)
        RP = skimage.measure.regionprops(np.uint8(self.minutiaeBif))
        WindowSize = 1  # --> For Bifurcation, the block size must be 3x3. Hence the window selected is 1
        for i in RP:
            (row, col) = np.int16(np.round(i['Centroid']))
            block = self._skel[row - WindowSize:row + WindowSize + 1, col - WindowSize:col + WindowSize + 1]
            angle = self.__computeAngle(block, 'Bifurcation')
            if(len(angle) == 3):
                FeaturesBif.append(MinutiaeFeature(row, col, angle, 1))

        FeaturesCross = []
        self.minutiaeCross = skimage.measure.label(self.minutiaeCross, connectivity=2)
        RP = skimage.measure.regionprops(np.uint8(self.minutiaeCross))
        WindowSize = 1  # --> For Crosspoint, the block size must be 3x3. Hence the window selected is 1
        for i in RP:
            (row, col) = np.int16(np.round(i['Centroid']))
            block = self._skel[row - WindowSize:row + WindowSize + 1, col - WindowSize:col + WindowSize + 1]
            angle = self.__computeAngle(block, 'Crosspoint')
            if(len(angle) == 4):
                FeaturesCross.append(MinutiaeFeature(row, col, angle, 2))

        return (FeaturesTerm, FeaturesBif, FeaturesCross)

    def extractMinutiaeFeatures(self, img):
        self.__skeletonize(img)

        self.__getTerminationBifurcationCrosspoint()

        self.__cleanMinutiae(img)

        FeaturesTerm, FeaturesBif, FeaturesCross = self.__performFeatureExtraction()

        (rows, cols) = self._skel.shape
        DispImg = np.zeros((rows, cols, 3), np.uint8)
        DispImg[:, :, 0] = 255 * self._skel
        DispImg[:, :, 1] = 255 * self._skel
        DispImg[:, :, 2] = 255 * self._skel

        return(DispImg, FeaturesTerm, FeaturesBif, FeaturesCross)

    def showResults(self, FeaturesTerm, FeaturesBif, FeaturesCross, isEndpoint=True, isBipoint=True):
        
        (rows, cols) = self._skel.shape
        DispImg = np.zeros((rows, cols, 3), np.uint8)
        DispImg[:, :, 0] = 255*self._skel
        DispImg[:, :, 1] = 255*self._skel
        DispImg[:, :, 2] = 255*self._skel
        half_side = 3

        if isEndpoint:
            for idx, curr_minutiae in enumerate(FeaturesTerm):
                col, row = curr_minutiae.locX, curr_minutiae.locY
                (rr, cc) = skimage.draw.rectangle_perimeter((col - half_side, row - half_side), (col + half_side, row + half_side))
                skimage.draw.set_color(DispImg, (rr, cc), (0, 0, 255))
                # draw orientation
                end_x = int(row + (half_side + 1) * np.cos(np.deg2rad(max(curr_minutiae.Orientation))))
                end_y = int(col - (half_side + 1) * np.sin(np.deg2rad(max(curr_minutiae.Orientation))))
                (rr, cc) = skimage.draw.line(col, row, end_y, end_x)
                skimage.draw.set_color(DispImg, (rr, cc), (0, 0, 255))

        if isBipoint:
            for idx, curr_minutiae in enumerate(FeaturesBif):
                col, row = curr_minutiae.locX, curr_minutiae.locY
                (rr, cc) = skimage.draw.rectangle_perimeter((col - half_side, row - half_side), (col + half_side, row + half_side))
                skimage.draw.set_color(DispImg, (rr, cc), (255, 0 , 0))
                # draw orientation
                end_x = int(row + (half_side + 1) * np.cos(np.deg2rad(max(curr_minutiae.Orientation))))
                end_y = int(col - (half_side + 1) * np.sin(np.deg2rad(max(curr_minutiae.Orientation))))
                (rr, cc) = skimage.draw.line(col, row, end_y, end_x)
                skimage.draw.set_color(DispImg, (rr, cc), (255, 0, 0))

            for idx, curr_minutiae in enumerate(FeaturesCross):
                col, row = curr_minutiae.locX, curr_minutiae.locY
                (rr, cc) = skimage.draw.rectangle_perimeter((col - half_side, row - half_side), (col + half_side, row + half_side))
                skimage.draw.set_color(DispImg, (rr, cc), (0, 255, 0))
                # draw orientation
                end_x = int(row + (half_side + 1) * np.cos(np.deg2rad(max(curr_minutiae.Orientation))))
                end_y = int(col - (half_side + 1) * np.sin(np.deg2rad(max(curr_minutiae.Orientation))))
                (rr, cc) = skimage.draw.line(col, row, end_y, end_x)
                skimage.draw.set_color(DispImg, (rr, cc), (0, 255, 0))
        
        # cv2.imshow('output', DispImg)
        # cv2.waitKey(0)
        return DispImg

    def saveResult(self, FeaturesTerm, FeaturesBif, FeaturesCross):
        (rows, cols) = self._skel.shape
        DispImg = np.zeros((rows, cols, 3), np.uint8)
        DispImg[:, :, 0] = 255 * self._skel
        DispImg[:, :, 1] = 255 * self._skel
        DispImg[:, :, 2] = 255 * self._skel
        # cv2.imwrite('result_line.png', DispImg)

        # for idx, curr_minutiae in enumerate(FeaturesTerm):
        #     row, col = curr_minutiae.locX, curr_minutiae.locY
        #     (rr, cc) = skimage.draw.circle_perimeter(row, col, 3)
        #     skimage.draw.set_color(DispImg, (rr, cc), (0, 0, 255))

        half_side = 3

        for idx, curr_minutiae in enumerate(FeaturesTerm):
            col, row = curr_minutiae.locX, curr_minutiae.locY
            (rr, cc) = skimage.draw.rectangle_perimeter((col - half_side, row - half_side), (col + half_side, row + half_side))
            skimage.draw.set_color(DispImg, (rr, cc), (0, 0, 255))
            # draw orientation
            end_x = int(row + (half_side + 1) * np.cos(np.deg2rad(max(curr_minutiae.Orientation))))
            end_y = int(col - (half_side + 1) * np.sin(np.deg2rad(max(curr_minutiae.Orientation))))
            (rr, cc) = skimage.draw.line(col, row, end_y, end_x)
            skimage.draw.set_color(DispImg, (rr, cc), (0, 0, 255))

        for idx, curr_minutiae in enumerate(FeaturesBif):
            col, row = curr_minutiae.locX, curr_minutiae.locY
            (rr, cc) = skimage.draw.rectangle_perimeter((col - half_side, row - half_side), (col + half_side, row + half_side))
            skimage.draw.set_color(DispImg, (rr, cc), (255, 0 , 0))
            # draw orientation
            end_x = int(row + (half_side + 1) * np.cos(np.deg2rad(max(curr_minutiae.Orientation))))
            end_y = int(col - (half_side + 1) * np.sin(np.deg2rad(max(curr_minutiae.Orientation))))
            (rr, cc) = skimage.draw.line(col, row, end_y, end_x)
            skimage.draw.set_color(DispImg, (rr, cc), (255, 0, 0))

        for idx, curr_minutiae in enumerate(FeaturesCross):
            col, row = curr_minutiae.locX, curr_minutiae.locY
            (rr, cc) = skimage.draw.rectangle_perimeter((col - half_side, row - half_side), (col + half_side, row + half_side))
            skimage.draw.set_color(DispImg, (rr, cc), (0, 255, 0))
            # draw orientation
            end_x = int(row + (half_side + 1) * np.cos(np.deg2rad(max(curr_minutiae.Orientation))))
            end_y = int(col - (half_side + 1) * np.sin(np.deg2rad(max(curr_minutiae.Orientation))))
            (rr, cc) = skimage.draw.line(col, row, end_y, end_x)
            skimage.draw.set_color(DispImg, (rr, cc), (0, 255, 0))

        cv2.imwrite('result.png', DispImg)

def extract_minutiae_features(img, isEndpoint, isBipoint, spuriousMinutiaeThresh=10, invertImage=False, showResult=False, saveResult=False):
    feature_extractor = FingerprintFeatureExtractor()
    feature_extractor.setSpuriousMinutiaeThresh(spuriousMinutiaeThresh)
    if (invertImage):
        img = 255 - img

    DispImg, FeaturesTerm, FeaturesBif, FeaturesCross = feature_extractor.extractMinutiaeFeatures(img)

    if (saveResult):
        feature_extractor.saveResult(FeaturesTerm, FeaturesBif, FeaturesCross)

    if(showResult):
        result = feature_extractor.showResults(FeaturesTerm, FeaturesBif, FeaturesCross, isEndpoint, isBipoint)

    return(DispImg, result, FeaturesTerm, FeaturesBif, FeaturesCross)
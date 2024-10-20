### built-in modules
import time
import math
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import os
from bio_modals.fingervein_data_loading import BasicDataset
import logging
#from bio_modals.unet import UNet
from PIL import Image
from skimage.morphology import skeletonize
from skimage import img_as_ubyte
import random
def skeleton(img):
    _, binary_img = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)
    binary_img = binary_img / 255.0
    skeleton = skeletonize(binary_img)
    skeleton = img_as_ubyte(skeleton)

    return skeleton

def smooth_image(skeleton_image):
    kernel_size = 3
    structuring_element = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    dilated_image = cv2.dilate(skeleton_image, structuring_element, iterations=3)
    smoothed_image = cv2.erode(dilated_image, structuring_element, iterations=1)
    return smoothed_image

### https://gist.github.com/neeru1207/dc30df52237d5c58ded47c43ed3dcf89
element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
def new_skel(image):
    skel = np.zeros(image.shape, np.uint8)

    ret, img = cv2.threshold(image, 127, 255, 0)
    # Repeat steps 2-4
    while True:
        # Step 2: Open the image
        open = cv2.morphologyEx(img, cv2.MORPH_OPEN, element)
        # Step 3: Substract open from the original image
        temp = cv2.subtract(img, open)
        # Step 4: Erode the original image and refine the skeleton
        eroded = cv2.erode(img, element)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded.copy()
        # Step 5: If there are no white pixels left ie.. the image has been completely eroded, quit the loop
        if cv2.countNonZero(img) == 0:
            break

    return skel


def extract_minutiae(image):
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    s_time = time.time()
    curve = draw_curve_vein(image)
    print('draw_curve_vein %.1fms' % ((time.time() - s_time) * 1000))

    s_time = time.time()
    line = skeleton(curve)
    print('skeleton %.1fms' % ((time.time() - s_time) * 1000))

    # s_time = time.time()
    # line = new_skel(image)
    # print('new_skel %.1fms' % ((time.time() - s_time) * 1000))

    s_time = time.time()
    smooth = smooth_image(line)
    #print('smooth_image %.1fms' % ((time.time() - s_time) * 1000))
    return smooth

def smooth_image2(skeleton_image):
    kernel_size = 3
    structuring_element = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    dilated_image = cv2.dilate(skeleton_image, structuring_element, iterations=3)
    #cv2.imshow('dilated_image', dilated_image)
    #cv2.waitKey(0)
    return dilated_image

def matching_image(image1, image2):
    # smooth1 = image1
    # smooth2 = image2
    matched = False
    best_score = -1  # Initialize the best score
    # best_angle = 0  # Initialize the best angle

    errors = 9999999999999999
    answer_min_loc = 0
    # answer_max_loc = 0
    # matching_scores = []
    dilated_image1 = image1
    dilated_image2 = image2
    #dilated_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    #dilated_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    #cv2.imshow('dilated_image1.jpg', dilated_image1)
    #cv2.imshow('dilated_image2.jpg', dilated_image2)
    best_angle = 0
    angle_range = range(-10, 10)

    for roi_x in range(70, 25, -5):
        # template_height, template_width = dilated_image2.shape
        image_height, image_width = dilated_image1.shape
        temp = dilated_image2[roi_x:(image_height) - (roi_x), roi_x: (image_width - roi_x)]

        for angle in [0.1 * i for i in angle_range]:
            rotated_template = temp.copy()
            M = cv2.getRotationMatrix2D((temp.shape[1] / 2, temp.shape[0] / 2), angle, 1)
            rotated_template = cv2.warpAffine(rotated_template, M, (temp.shape[1], temp.shape[0]))
            st = time.time()
            result = cv2.matchTemplate(dilated_image1, rotated_template, cv2.TM_SQDIFF)
            #print('%.1fms'%(time.time()-st)*1000)
            # result_norm = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            if errors > min_val:
                errors = min_val
                answer_min_loc = min_loc

                th, tw = temp.shape[:2]
                # print('minv:', min_val)
                # print('minloc:', min_loc)
                # print('tw,th', tw, th)
                # print('angle', angle)
                target1 = dilated_image1[answer_min_loc[1]: answer_min_loc[1] + th,
                          answer_min_loc[0]: answer_min_loc[0] + tw]
                answer_temp = rotated_template.copy()
                dst = cv2.bitwise_or(target1, answer_temp)
                # temp1 = cv2.bitwise_and(target1, answer_temp)

                #cv2.imshow('temp1', temp1)
                # print("temp1 : ", cv2.countNonZero((cv2.bitwise_and(target1, answer_temp))))
                # print("total : ", (cv2.countNonZero(target1)))

                # print("전체 정맥 픽셀 중 매칭 픽셀중 정맥 비율",
                #       (cv2.countNonZero(cv2.bitwise_and(target1, answer_temp)) / (cv2.countNonZero(dst))) * 100)
                # norm_errors = (errors / (cv2.countNonZero(dst) * 255 ** 2))
                score = (cv2.countNonZero(cv2.bitwise_and(target1, answer_temp)) / (cv2.countNonZero(dst))) * 100
                if best_score < score:
                    best_score = score
                    best_angle = angle
                    best_min_loc = min_loc
                    best_roi_x = roi_x
                    # np.stack([image,])
                    zero2 = np.zeros_like(dilated_image2)
                    zero2[answer_min_loc[1]:answer_min_loc[1] + th,
                    answer_min_loc[0]:answer_min_loc[0] + tw] = answer_temp.copy()
            else:
                pass
        adjust_score = best_score - abs(5.5077 * best_angle) + 1.0005 * best_roi_x + (-0.4359) * abs(
            best_min_loc[0] - best_roi_x) + (-0.7564) * abs(best_min_loc[1] - best_roi_x)
        if adjust_score < 80:


            matched = False
            break
    # target1 = image1[answer_min_loc[1]: answer_min_loc[1] + th, answer_min_loc[0]: answer_min_loc[0] + tw]
    #target1 = cv2.cvtColor(target1, cv2.COLOR_BGR2GRAY)
    #print('Best angle :', best_angle)
    #
    #print('Margin :', best_roi_x)
    #print('x만큼 이동 :', best_min_loc[0] - best_roi_x)
    # x_move = best_min_loc[0] - best_roi_x
    # y_move = best_min_loc[1] - best_roi_x
    #print('y만큼 이동 :', best_min_loc[1] - best_roi_x)
    adjust_score = best_score - abs(5.5077 * best_angle) + 1.0005 * best_roi_x + (-0.4359) * abs(
        best_min_loc[0] - best_roi_x) + (-0.7564) * abs(best_min_loc[1] - best_roi_x)
    #print('Score : ', best_score)
    print('Adjust_score', adjust_score)
    if adjust_score > 80:
        matched = True

    if adjust_score < 0:
        adjust_score = 0

    return adjust_score, matched



def mask_to_image(mask: np.ndarray, mask_values):
    if isinstance(mask_values[0], list):
        out = np.zeros((mask.shape[-2], mask.shape[-1], len(mask_values[0])), dtype=np.uint8)
    elif mask_values == [0, 1]:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=bool)
    else:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=np.uint8)
    if mask.ndim == 3:
        mask = np.argmax(mask, axis=0)
    for i, v in enumerate(mask_values):
        out[mask == i] = v
    return Image.fromarray(out)

def create_segmentation(fingervein):
    ori_img = fingervein
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    net = UNet(n_channels=1, n_classes=2, bilinear=False)
    img = Image.fromarray(fingervein)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Loading model {model}')
    logging.info(f'Using device {device}')

    net.to(device=device)
    state_dict = torch.load(model, map_location=device)
    mask_values = state_dict.pop('mask_values', [0, 1])
    net.load_state_dict(state_dict)

    mask = predict_img(net=net,
                       full_img=img,
                       scale_factor=0.3,
                       out_threshold=0.3,
                       device=device)
    result = mask_to_image(mask, mask_values)
    mask_np = np.array(result, np.uint8)

    cv2.imshow('modify_mask', mask_np)
    cv2.waitKey(0)
    return mask_np

def remove_noise(segmentation_map, kernel_size=(10, 10)):
    # 모폴로지 연산을 위한 커널 생성
    kernel = np.ones(kernel_size, np.uint8)
    # 열림 연산으로 작은 객체나 노이즈 제거
    opening = cv2.morphologyEx(segmentation_map, cv2.MORPH_OPEN, kernel)
    # 닫힘 연산으로 작은 구멍이나 내부의 검은 점들 제거
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    #cv2.imshow('closing',closing )
    #cv2.waitKey(0)
    return closing
def draw_curve_vein(fingervein):
    mask = np.ones(fingervein.shape, np.uint8)  # Locus space
    result = make_skeltonimage(fingervein, mask, 8)
    _, max_val = cv2.minMaxLoc(result)[:2]
    skelton_result = ((result * 255.0 / max_val) * 20).astype(np.uint8)
    return skelton_result

def _meshgrid(xgv, ygv):
    x = np.outer(np.ones_like(ygv), xgv)
    y = np.outer(ygv, np.ones_like(xgv))
    return x, y
def _conv(src, kernel):
    anchor = (kernel.shape[1] - kernel.shape[1] // 2 - 1, kernel.shape[0] - kernel.shape[0] // 2 - 1)
    flipped = cv2.flip(kernel, 0)
    result = cv2.filter2D(src, -1, flipped, anchor=anchor, borderType=cv2.BORDER_REPLICATE)
    return result

def make_skeltonimage(_src, _mask, sigma):
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

def make_condition_image(img):
    dst = draw_curve_vein(img)
    print(dst.shape)

    dst = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)
    #kernel_size = 3
    #structuring_element = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    #dst = cv2.erode(dst, structuring_element, iterations=0.5)
    dst[dst<50] = 0
    dst[dst>50] = 255
    dst_return = dst
    #cv2.imshow('dst', dst)
    #cv2.imwrite('dst_modify.png', dst)
    #cv2.waitKey(0)
    #src = create_segmentation(img)
    folder_path = '../dataset/VERA_segmentation_padding'
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    random_file = random.choice(files)
    image_path = os.path.join(folder_path, random_file)
    src = cv2.imread(image_path)
    #src = cv2.cvtColor(src, cv2.COLOR_GRAY2BGR)

    if src.shape[:2] != dst.shape[:2]:
        raise ValueError("Image sizes do not match.")

        # 이미지 처리
    for y in range(src.shape[0]):
        for x in range(src.shape[1]):
            if np.all(src[y, x] == [255, 255, 255]):  # 마스크 이미지의 흰색 픽셀이면
                src[y, x] = np.array([255, 0, 0]) # 해당 픽셀을 파란색으로 변경 (BGR 순서)
    for y in range(dst.shape[0]):
        for x in range(dst.shape[1]):
            if np.all(dst[y, x] > 3):  # 마스크 이미지의 흰색 픽셀이면
                dst[y, x] = np.array([0, 255, 255], dtype=np.uint8)
    result = cv2.add(dst, src)
    for y in range(result.shape[0]):
        for x in range(result.shape[1]):
            if np.all(result[y, x] == [0, 255, 255]):  # 노란색
                result[y, x] = np.array([0, 0, 0], dtype=np.uint8)
            if np.all(result[y, x] == [255, 255, 255]):
                result[y, x] = np.array([0, 255, 255], dtype=np.uint8)
    return result, dst_return


def predict_img(net,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.5):
    net.eval()
    img = torch.from_numpy(BasicDataset.preprocess(None, full_img, scale_factor, is_mask=False))
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)
    with torch.no_grad():
        output = net(img).cpu()
        output = F.interpolate(output, (full_img.size[1], full_img.size[0]), mode='bilinear')
        if net.n_classes > 1:
            mask = output.argmax(dim=1)
        else:
            mask = torch.sigmoid(output) > out_threshold
    return mask[0].long().squeeze().numpy()


def make_condition_image_many(img):
    results = []
    dst = img
    #cv2.imshow('img', img)
    #cv2.waitKey(0)
    for _ in range(5):

        folder_path = '../../Dataset/VERA_segmentation_padding'
        files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
        random_file = random.choice(files)
        image_path = os.path.join(folder_path, random_file)
        src = cv2.imread(image_path)

        if src.shape[:2] != dst.shape[:2]:
            raise ValueError("Image sizes do not match.")

            # 이미지 처리
        for y in range(src.shape[0]):
            for x in range(src.shape[1]):
                if np.all(src[y, x] == [255, 255, 255]):  # 마스크 이미지의 흰색 픽셀이면
                    src[y, x] = np.array([255, 0, 0]) # 해당 픽셀을 파란색으로 변경 (BGR 순서)
        for y in range(dst.shape[0]):
            for x in range(dst.shape[1]):
                if np.all(dst[y, x] > 3):  # 마스크 이미지의 흰색 픽셀이면
                    dst[y, x] = np.array([0, 255, 255], dtype=np.uint8)
        result = cv2.add(dst, src)
        for y in range(result.shape[0]):
            for x in range(result.shape[1]):
                if np.all(result[y, x] == [0, 255, 255]):  # 노란색
                    result[y, x] = np.array([0, 0, 0], dtype=np.uint8)
                if np.all(result[y, x] == [255, 255, 255]):
                    result[y, x] = np.array([0, 255, 255], dtype=np.uint8)


        x_offset = random.randint(-5, 5)
        y_offset = random.randint(-15, 15)
        height, width, channels = result.shape
        # 변환 행렬 생성
        M = np.float32([[1, 0, x_offset], [0, 1, y_offset]])
        shifted_image = cv2.warpAffine(result, M, (width, height))
        results.append(shifted_image)
        print( "make_condition_image")
    return results[0], results[1], results[2], results[3]



def extract_Segmentation(img):
    pass

def unit_test_match(obj):
    print('######################### Unit Test 1 - grayscale image input #########################')
    img1 = cv2.imread("../unit_test_data/Fingerprint/093/L3_03.BMP", cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread("../unit_test_data/Fingerprint/093/L3_04.BMP", cv2.IMREAD_GRAYSCALE)
    matched, matching_score, quality1, quality2 = obj.match_using_images(img1, img2)
    print(matched, matching_score, quality1, quality2)
    print('######################### Unit Test 2 - color image input #########################')
    img1 = cv2.imread("../unit_test_data/Fingerprint/093/L3_03.BMP", cv2.IMREAD_COLOR)
    img2 = cv2.imread("../unit_test_data/Fingerprint/093/L3_04.BMP", cv2.IMREAD_COLOR)
    matched, matching_score, quality1, quality2 = obj.match_using_images(img1, img2)
    print(matched, matching_score, quality1, quality2)
    print('######################### Unit Test 3 - file path input #########################')
    img1 = "../unit_test_data/Fingerprint/093/L3_03.BMP"
    img2 = "../unit_test_data/Fingerprint/093/L3_04.BMP"
    matched, matching_score, quality1, quality2 = obj.match_using_images(img1, img2)
    print(matched, matching_score, quality1, quality2)
    print('######################### Unit Test 4 - weired path input #########################')
    img1 = r'C:\weired\path'
    img2 = r'C:\weired\path2'
    try:
        matched, matching_score, quality1, quality2 = obj.match_using_images(img1, img2)
        print(matched, matching_score, quality1, quality2)
    except Exception as e:
        print(type(e))
    print()


def unit_test_match_using_filelist(obj):
    print('######################### Unit Test 1 - filelist1 #########################')
    filelist1 = [
        "../unit_test_data/Fingerprint/093/L3_03.BMP",
        "../unit_test_data/Fingerprint/094/R3_01.BMP",
        "../unit_test_data/Fingerprint/227/L2_04.BMP",
    ]
    results, qualities1, qualities2 = obj.match_using_filelist(filelist1)
    print(results, qualities1, qualities2)

    print('################### Unit Test 2 - filelist1 and filelist2 ###################')
    filelist2 = [
        "../unit_test_data/Fingerprint/093/L3_04.BMP",
        "../unit_test_data/Fingerprint/094/R3_02.BMP",
        "../unit_test_data/Fingerprint/227/L2_05.BMP",
    ]
    results, qualities1, qualities2 = obj.match_using_filelist(filelist1, filelist2)
    print(results, qualities1, qualities2)

    print('######################### Unit Test 3 - file error #########################')
    filelist3 = [r"C:\weired\path", r"C:\weired\path2"]
    try:
        results, qualities1, qualities2 = obj.match_using_filelist(filelist3)
        print(results, qualities1, qualities2)
    except Exception as e:
        print(type(e))
    print()


if __name__ == '__main__':
    test1 = cv2.imread(r"C:\Users\CVlab\Documents\01_2023\01_2023_BiosyntheticData\Dataset\Train_fingervein\VERA-FV-full-bf\105_L_1.bmp", cv2.IMREAD_GRAYSCALE)
    test2 = cv2.imread(r"C:\Users\CVlab\Documents\01_2023\01_2023_BiosyntheticData\Dataset\Train_fingervein\VERA-FV-full-bf\105_L_2.bmp", cv2.IMREAD_GRAYSCALE)
    #test = "../img_TH_EN.bmp"
    #create_segmentation(test)
    #draw_curve_vein(test)
    #create_segmentation(test)
    threshold_value = 128

    # s_time = time.time()
    smooth1 = extract_minutiae(test1)
    # print('extract_minutiae %.1fms' % ((time.time() - s_time) * 1000))
    smooth2 = extract_minutiae(test2)
    #smooth2 = smooth_image2(smooth)

    s_time = time.time()
    matching_score, matched = matching_image(smooth1, smooth2)
    print('matching_image %.1fms' % ((time.time() - s_time) * 1000))
    print(matching_score)
    #_, binary_image1 = cv2.threshold(skel, threshold_value, 255, cv2.THRESH_BINARY)
    #smooth_image2(skel)

    #make_condition_image(test)
    # unit_test_match(obj)
    # unit_test_match_using_filelist(obj)

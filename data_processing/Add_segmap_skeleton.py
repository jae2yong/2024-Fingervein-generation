from PIL import Image
import cv2
import numpy as np
import os
# 이미지 파일 경로 설정

src_path = "VERA-Segmentation/"
seg_file = os.listdir(src_path)
dst_path = "VERA-Skeleton/"
dst_file = os.listdir(dst_path)
#print(seg_file)
#print(dst_file)
# for seg_files in seg_file:
#     seg_file_name = seg_files[:-4]
    #print(file_name)
for skel_files in dst_file:
    skel_name = skel_files[:-4]
    print("skel_files : ", skel_files)
    for seg_files in seg_file:
        seg_file_name = seg_files[:-7]
        print("seg_files : ", seg_files)
        if skel_name == seg_file_name:
            src = cv2.imread(src_path + seg_file_name +'.bmp')
            dst = cv2.imread(dst_path + skel_name +'.jpg')
            #print(src_path + seg_file_name +'.bmp')
            # 두 이미지의 크기가 같은지 확인
            if src.shape[:2] != dst.shape[:2]:
                raise ValueError("Image sizes do not match.")

            # 이미지 처리
            for y in range(src.shape[0]):
                for x in range(src.shape[1]):
                    if np.all(src[y, x] == [255, 255, 255]):  # 마스크 이미지의 흰색 픽셀이면
                        src[y, x] = np.array([255, 0, 0], dtype=np.uint8)
                        # 해당 픽셀을 파란색으로 변경 (BGR 순서)
            #cv2.imshow('src', src)
            #cv2.waitKey(0)
            for y in range(dst.shape[0]):
                for x in range(dst.shape[1]):
                    if np.all(dst[y, x] > 9):  # 마스크 이미지의 흰색 픽셀이면
                        dst[y, x] = np.array([0, 255, 255], dtype=np.uint8)  # 해당 픽셀을 파란색으로 변경 (BGR 순서)


            result = cv2.add(dst, src)

            for y in range(result.shape[0]):
                for x in range(result.shape[1]):
                    if np.all(result[y, x] == [0, 255, 255]):  # 노란색
                        result[y, x] = np.array([0, 0, 0], dtype=np.uint8)  # 해당 픽셀을 파란색으로 변경 (BGR 순서)
                    if np.all(result[y, x] == [255, 255, 255]):
                        result[y, x] = np.array([0, 255, 255], dtype=np.uint8)
            
            #cv2.imwrite('test.jpg', result)
        # cv2.imshow('Visualization2', src)
            print("create files :" + str(skel_name)+'.jpg')
            cv2.imwrite(str(skel_name)+'.jpg', result)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
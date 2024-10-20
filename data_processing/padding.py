import os
from PIL import Image, ImageOps

def add_padding_to_image(input_path, output_path, target_width=1600, target_height=900):
    # 이미지 열기
    img = Image.open(input_path)
    
    # 현재 이미지 크기 가져오기
    width, height = img.size

    # 필요한 패딩 계산
    left_padding = (target_width - width) // 2
    top_padding = (target_height - height) // 2
    right_padding = target_width - width - left_padding
    bottom_padding = target_height - height - top_padding

    # 이미지에 패딩 추가
    padded_img = ImageOps.expand(img, (left_padding, top_padding, right_padding, bottom_padding), fill=(255, 255, 255))

    # 결과 이미지 저장
    padded_img.save(output_path)
    print(f"이미지가 성공적으로 저장되었습니다: {output_path}")

def process_images_in_folder(input_folder, output_folder, target_width=1600, target_height=900):
    # 입력 폴더의 모든 파일에 대해 반복
    for filename in os.listdir(input_folder):
        # 이미지 파일 경로 설정
        input_path = os.path.join(input_folder, filename)

        # 파일이 이미지 파일인지 확인
        if os.path.isfile(input_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            # 출력 파일 경로 설정
            output_path = os.path.join(output_folder, filename)

            # 이미지에 패딩 추가
            add_padding_to_image(input_path, output_path, target_width, target_height)

# 사용 예시
input_folder = r'C:\Users\CVlab\Documents\01_2023\01_2023_BiosyntheticData\08_가상데이터샘플및최종데이터\20240617_생성지정맥_1600X900'  # 입력 이미지 폴더 경로
output_folder = r'C:\Users\CVlab\Documents\01_2023\01_2023_BiosyntheticData\08_가상데이터샘플및최종데이터\20240617_생성지정맥_1600X900_2'  # 출력 이미지 폴더 경로

# 출력 폴더가 없으면 생성
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

process_images_in_folder(input_folder, output_folder)

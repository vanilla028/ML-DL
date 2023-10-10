import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import torch
import numpy as np

def augment_and_save(image_path, output_path):
    # 이미지를 NumPy 배열로 읽어옵니다.
    image = cv2.imread(image_path)
    
    # Albumentations 변환 파이프라인 정의
    train_transform = A.Compose([
        A.GaussianBlur(blur_limit=(25, 25), p=1),
        ToTensorV2()
    ])
    
    # Albumentations를 사용하여 이미지 증강 및 텐서로 변환
    augmented = train_transform(image=image)
    augmented_image = augmented['image']
    
    # 텐서를 NumPy 배열로 변환
    augmented_image = augmented_image.permute(1, 2, 0).cpu().numpy()
    
    # 변환된 이미지를 저장
    cv2.imwrite(output_path, augmented_image)

# 원본 이미지 경로와 저장할 경로 설정
image_path = 'scissors.jpg'
output_path = 'augmented_image.jpg'

# 이미지 증강 및 저장
augment_and_save(image_path, output_path)
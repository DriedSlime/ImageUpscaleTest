from realesrgan import RealESRGANer   # 업스케일링에 사용할 딥러닝 기반 모델
from PIL import Image
import torch
import numpy as np

def upscale_image(input_path, output_path, scale=4):
    
    # 지포스 그래픽카드가 있으면 CUDA 사용용
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 모델 초기화
    model = RealESRGANer(
        scale=scale,
        model_path='weights/RealESRGAN_x4plus.pth', 
        device=device
        )
    model.load_weights(f'RealESRGAN_x{scale}.path', download=True)

    # 이미지 로드
    image = Image.open(input_path).convert('RGB')
    img_array = np.array(image)

    # 업스케일 처리
    sr_image, _ = model.enhance(img_array, outscale=scale)

    # 결과
    sr_image = Image.fromarray(sr_image)
    sr_image.save(output_path)
    print(f"Saved upscaled image to {output_path}")

if __name__ == '__main__':
    input_image_path = './image.png'
    output_image_path = './upscaled.jpg'

    upscale_image(input_image_path, output_image_path, scale=4)
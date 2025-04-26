from realesrgan import RealESRGANer
from realesrgan.archs.rrdbnet_arch import RRDBNet
from PIL import Image
import torch
import numpy as np

def upscale_image(input_path, output_path, scale=4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 모델 만들기
    model = RRDBNet(
        num_in_ch=3, num_out_ch=3,
        num_feat=64, num_block=23,
        num_grow_ch=32, scale=scale
    )

    # RealESRGANer 만들기
    upsampler = RealESRGANer(
        scale=scale,
        model_path='weights/RealESRGAN_x4plus.pth',
        model=model,
        device=device,
        tile=0,
        tile_pad=10,
        pre_pad=0
    )

    # 이미지 로드
    image = Image.open(input_path).convert('RGB')
    img_array = np.array(image)

    # 업스케일링
    output, _ = upsampler.enhance(img_array, outscale=scale)

    # 저장
    output_image = Image.fromarray(output)
    output_image.save(output_path)
    print(f"Saved upscaled image to {output_path}")

if __name__ == '__main__':
    input_image_path = './image.png'  # 입력 이미지 경로
    output_image_path = './upscaled_image.png'  # 출력 이미지 경로

    upscale_image(input_image_path, output_image_path, scale=4)  # scale 값 4로 설정

import torch
from PIL import Image
from RealESRGAN import RealESRGAN
import os

def upscale_image(input_path, output_path, scale=4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 출력 폴더 자동 생성
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    model = RealESRGAN(device, scale=scale)
    model.load_weights(f'weights/RealESRGAN_x{scale}.pth', download=True)

    img = Image.open(input_path).convert('RGB')
    sr_img = model.predict(img)

    # 저장
    sr_img.save(output_path)
    print(f'업스케일 완료! 저장 경로: {output_path}')
    return output_path

# 실행 결과물 바로 폴더에 저장
upscale_image("input.jpg", "output/input_x4.jpg", scale=4)

# if __name__ == '__main__':
#     # cmd로 인자 입력 받기
#     parser = argparse.ArgumentParser()
#     parser.add_argument('input', type=str, help='입력 이미지 경로')
#     parser.add_argument('output', type=str, help='출력 이미지 경로')
#     parser.add_argument('--scale', type=int, choices=[2, 4, 8], default=4, help='업스케일 배율')
#     args = parser.parse_args()

#     upscale_image(args.input, args.output, args.scale)

import torch
from PIL import Image
from RealESRGAN import RealESRGAN

def upscale_image(input_path, output_path, scale=4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = RealESRGAN(device, scale=scale)
    model.load_weights(f'weights/RealESRGAN_x{scale}.pth', download=True)

    img = Image.open(input_path).convert('RGB')

    sr_img = model.predict(img)
    sr_img.save(output_path)

    print(f'업스케일 완료! 저장 경로: {output_path}')

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str, help='입력 이미지 경로')
    parser.add_argument('output', type=str, help='출력 이미지 경로')
    parser.add_argument('--scale', type=int, choices=[2, 4, 8], default=4, help='업스케일 배율')
    args = parser.parse_args()

    upscale_image(args.input, args.output, args.scale)

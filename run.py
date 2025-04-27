import os
import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor, to_pil_image
from basicsr.archs.rrdbnet_arch import RRDBNet

def load_model(scale, device):
    model = RRDBNet(
        num_in_ch=3, num_out_ch=3, num_feat=64,
        num_block=23, num_grow_ch=32, scale=scale
    )

    model_path = f'RealESRGAN_x{scale}.pth'
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"모델 파일이 없습니다: {model_path}")

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint, strict=True)
    model = model.to(device)
    model.eval()

    return model

def upscale_image(input_path, output_path, scale=4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(scale, device)

    img = Image.open(input_path).convert('RGB')
    img_tensor = to_tensor(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output_tensor = model(img_tensor)
    
    output_tensor = output_tensor.squeeze(0).clamp(0, 1).cpu()
    output_img = to_pil_image(output_tensor)

    output_img.save(output_path)
    print(f'✅ 업스케일 완료! 저장 경로: {output_path}')

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str, help='입력 이미지 경로')
    parser.add_argument('output', type=str, help='출력 이미지 경로')
    parser.add_argument('--scale', type=int, default=4, help='업스케일 배율 (기본 4)')
    args = parser.parse_args()

    upscale_image(args.input, args.output, args.scale)

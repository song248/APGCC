import os
import cv2
import torch
import argparse
import numpy as np
import random

# Custom Modules from your project
from models import build_model
from config import cfg, merge_from_file, merge_from_list

def parse_args():
    parser = argparse.ArgumentParser('Image Inference for APGCC')
    parser.add_argument('-c', '--config_file', type=str, default="./configs/SHHA_test.yml",
                        help='Path to config file')
    parser.add_argument('--image', type=str, required=True,
                        help='Path to image file for inference')
    parser.add_argument('--save_image', type=str, default=None,
                        help='Path to save the output image (optional)')
    parser.add_argument('opts', help='Overwrite config options from command line',
                        default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.config_file != "":
        merge_from_file(cfg, args.config_file)
    merge_from_list(cfg, args.opts)
    cfg.config_file = args.config_file
    return args, cfg

def setup_seed(seed):
    if seed is not None:
        g = torch.Generator()
        g.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = False
        torch.use_deterministic_algorithms(True, warn_only=True)
        os.environ['PYTHONHASHSEED'] = str(seed)

def image_inference(cfg, image_path, save_image_path=None):
    device = torch.device('cuda')
    
    # Build model and load pretrained weights
    model = build_model(cfg=cfg, training=False)
    model.to(device)
    model.eval()
    pretrained_dict = torch.load(cfg.TEST.WEIGHT, map_location='cpu')
    model_dict = model.state_dict()
    # Load only matching keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    
    # Read input image
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Cannot read image file:", image_path)
        return
    
    print('')
    print(f"Original Image Shape: {img.shape}")
    target_size = (1024, 768)
    img_resized = cv2.resize(img, target_size)
    print(f"Resized Image Shape: {img_resized.shape}")
    print('')

    # Preprocess image: BGR -> RGB, convert to float tensor, scale to [0, 1] and add batch dimension
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    input_tensor = torch.from_numpy(img_rgb).float() / 255.0
    input_tensor = input_tensor.permute(2, 0, 1).unsqueeze(0).to(device)
    
    # Model inference
    with torch.no_grad():
        output = model(input_tensor)
    
    # Output structure check
    print(f"Output Keys: {output.keys()}")

    # Use 'offset' as density map instead of 'pred_points'
    if 'offset' in output:
        density_map = output['offset'].detach().cpu().squeeze(0).numpy()
    else:
        raise ValueError("No 'offset' key in model output. Cannot extract density map.")

    print(f"Density Map Shape: {density_map.shape}")
    print(f"Density Map Max Value: {density_map.max()}")
    print(f"Density Map Sum: {density_map.sum()}")

    if density_map.max() > 0:
        density_map = density_map / density_map.max()  # Scale to [0, 1]
        
    """
    with torch.no_grad():
        output = model(input_tensor)

    print(f"Output Type: {type(output)}")
    print(f"Output Keys: {output.keys()}") 
    
    # density_map = output[0].detach().cpu().squeeze(0).numpy()
    if isinstance(output, dict):
        if 'pred_points' in output:
            density_map = output['pred_points'].detach().cpu().squeeze(0).numpy()
            print(f"Density Map Shape: {density_map.shape}")  # 밀도 맵 크기 확인
            print(f"Density Map Max Value: {density_map.max()}")  # 최대값 확인
            print(f"Density Map Sum: {density_map.sum()}")  # 전체 합 확인
        else:
            raise ValueError(f"Unexpected output keys: {output.keys()}")
    else:
        density_map = output[0].detach().cpu().squeeze(0).numpy()
    """

    pred_count = density_map.sum()
    
    # 결과 오버레이: 원본 이미지에 예측 count 출력
    output_img = img_resized.copy()  # Resized 이미지 사용
    text = f"Count: {pred_count:.2f}"
    cv2.putText(output_img, text, (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # 결과를 output 폴더에 자동 저장
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)  # 폴더 없으면 생성
    filename = os.path.basename(image_path)  # 파일명 추출
    save_path = os.path.join(output_dir, f"result_{filename}")  # 저장 경로
    cv2.imwrite(save_path, output_img)

    print(f"Inference complete. Predicted count: {pred_count:.2f}")
    print(f"Saved output image to {save_path}")
def main():
    args, cfg = parse_args()
    print('<cfg>')
    print(cfg)
    print(cfg.MODEL)

    os.environ["CUDA_VISIBLE_DEVICES"] = '{}'.format(cfg.GPU_ID)
    setup_seed(cfg.SEED)
    image_inference(cfg, args.image, args.save_image)

if __name__ == '__main__':
    main()

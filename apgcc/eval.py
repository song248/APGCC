import os
import glob
import pandas as pd
from tqdm import tqdm
import numpy as np
import cv2
import torch
import argparse

# 커스텀 모델 및 설정 관련 모듈 임포트
from models import build_model
from config import cfg, merge_from_file, merge_from_list

# 디렉토리 관련 설정 (assets/config.py에 정의된 변수 사용)
from assets.config import SOURCE_DIR, PRED_DIR, LABEL_DIR

def parse_args():
    parser = argparse.ArgumentParser('Evaluation with custom model')
    # 기본 config 파일 경로를 "./configs/SHHA_test.yml"로 지정
    parser.add_argument('-c', '--config_file', type=str, default="./configs/SHHA_test.yml",
                        help='Path to config file')
    parser.add_argument('opts', help='Overwrite config options from command line',
                        default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    merge_from_file(cfg, args.config_file)
    merge_from_list(cfg, args.opts)
    cfg.config_file = args.config_file
    return args, cfg

def load_custom_model(cfg):
    # cfg.TEST.WEIGHT 값 확인
    print("Using pretrained weight file:", cfg.TEST.WEIGHT)
    if not cfg.TEST.WEIGHT or cfg.TEST.WEIGHT.strip() == "":
        raise ValueError("cfg.TEST.WEIGHT is empty. "
                         "Please check your config file ({}). "
                         "It must specify the path to the pretrained weight.".format(cfg.config_file))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model(cfg=cfg, training=False)
    model.to(device)
    model.eval()
    
    pretrained_dict = torch.load(cfg.TEST.WEIGHT, map_location='cpu')
    model_dict = model.state_dict()
    # 모델에 해당하는 키만 업데이트
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    return model, device

def custom_model_inference(model, device, image_path):
    """
    이미지 경로를 받아 커스텀 모델 추론 후, density map의 합(예측 count)을 반환합니다.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Error: Cannot read image file: {image_path}")
    
    # (1024, 768)로 리사이즈
    target_size = (1024, 768)
    img_resized = cv2.resize(img, target_size)
    
    # 전처리: BGR -> RGB, 텐서 변환, [0, 1] 스케일링, 차원 변경
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    input_tensor = torch.from_numpy(img_rgb).float() / 255.0
    input_tensor = input_tensor.permute(2, 0, 1).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(input_tensor)
    
    # 'offset' 키로 density map 추출
    if 'offset' in output:
        density_map = output['offset'].detach().cpu().squeeze(0).numpy()
    else:
        raise ValueError("No 'offset' key in model output. Cannot extract density map.")
    
    if density_map.max() > 0:
        density_map = density_map / density_map.max()
    
    pred_count = density_map.sum()
    return pred_count

def evaluate_images(cfg):
    """
    여러 CSV 파일에 대해 각 이미지의 예측 count를 수행하고 결과를 CSV 파일로 저장합니다.
    """
    model, device = load_custom_model(cfg)
    os.makedirs(PRED_DIR, exist_ok=True)
    
    csv_files = glob.glob(os.path.join(LABEL_DIR, "*.csv"))
    print(f"{len(csv_files)}개의 CSV 파일을 찾았습니다.")
    
    for csv_file in csv_files:
        folder_name = os.path.basename(csv_file).replace('.csv', '')
        try:
            df = pd.read_csv(csv_file)
            print(f"  CSV 파일 로드됨: {csv_file} (행 수: {len(df)})")
        except Exception as e:
            print(f"  CSV 파일 로드 오류: {e}")
            continue
        
        image_folder = os.path.join(SOURCE_DIR, folder_name)
        if not os.path.exists(image_folder):
            # VL_로 시작하면 VS_로 변환
            if folder_name.startswith('VL_'):
                alt_folder_name = 'VS_' + folder_name[3:]
                alt_image_folder = os.path.join(SOURCE_DIR, alt_folder_name)
                if os.path.exists(alt_image_folder):
                    image_folder = alt_image_folder
                    print(f"  VL_을 VS_로 변환하여 폴더 찾음: {alt_folder_name}")
            # VS_로 시작하면 VL_로 변환
            elif folder_name.startswith('VS_'):
                alt_folder_name = 'VL_' + folder_name[3:]
                alt_image_folder = os.path.join(SOURCE_DIR, alt_folder_name)
                if os.path.exists(alt_image_folder):
                    image_folder = alt_image_folder
                    print(f"  VS_를 VL_로 변환하여 폴더 찾음: {alt_folder_name}")
        
        if not os.path.exists(image_folder):
            print(f"  이미지 폴더를 찾을 수 없음: {image_folder}")
            continue
        
        df['pred'] = np.nan
        processed_count = 0
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="이미지 처리 중"):
            image_name = row['image_name']
            image_path = os.path.join(image_folder, image_name)
            
            if not os.path.exists(image_path):
                print(f"  이미지를 찾을 수 없음: {image_path}")
                continue
            try:
                pred_count = custom_model_inference(model, device, image_path)
                df.at[idx, 'pred'] = pred_count
                processed_count += 1
            except Exception as e:
                print(f"  이미지 처리 오류 ({image_name}): {e}")
        
        output_file = os.path.join(PRED_DIR, os.path.basename(csv_file))
        df.to_csv(output_file, index=False)
        print(f"  처리 완료: {processed_count}/{len(df)} 이미지, 결과 저장됨: {output_file}")
    
    print("\n모든 이미지 처리 및 예측이 완료되었습니다.")

if __name__ == '__main__':
    args, cfg = parse_args()
    evaluate_images(cfg)

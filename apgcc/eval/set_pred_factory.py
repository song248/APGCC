import os
import glob
import pandas as pd
from tqdm import tqdm
import numpy as np
from abc import ABC, abstractmethod
from ultralytics import YOLO
from assets.config import PERSON_DET, SOURCE_DIR, PRED_DIR, LABEL_DIR

# 기본 모델 인터페이스
class DetectionModel(ABC):
    @abstractmethod
    def load_model(self, model_path):
        """모델을 로드하는 메서드"""
        pass
    
    @abstractmethod
    def predict(self, image_path, **kwargs):
        """이미지에서 사람 수를 예측하는 메서드"""
        pass

# YOLO 모델 어댑터
class YOLOAdapter(DetectionModel):
    def __init__(self):
        self.model = None
    
    def load_model(self, model_path):
        self.model = YOLO(model_path)
        return self
    
    def predict(self, image_path, **kwargs):
        conf = kwargs.get('conf', 0.15)
        classes = kwargs.get('classes', [0])
        
        results = self.model.predict(image_path, conf=conf, classes=classes)
        
        for result in results:
            return result.boxes.shape[0]
        return 0 
# ClipEBC 모델 어댑터
class ClipEBCAdapter(DetectionModel):
    def __init__(self):
        self.model = None
    
    def load_model(self, model_path=None):
        from CLIP_EBC.custom.clip_ebc import ClipEBC
        self.model = ClipEBC()
        return self
    
    def predict(self, image_path, **kwargs):
        count = self.model.predict(image_path)
        return count

# 모델 팩토리
class CrowdModelEval:
    @staticmethod
    def get_model(model_type, model_path=PERSON_DET):
        if model_type.lower() == 'yolo':
            return YOLOAdapter().load_model(model_path)
        elif model_type.lower() == 'clip_ebc':
            return ClipEBCAdapter().load_model(model_path)
        else:
            raise ValueError(f"지원하지 않는 모델 유형: {model_type}")

# 개선된 평가 함수
def evaluate_images(model_type='yolo', model_path=None, **kwargs):
    """
    이미지 평가를 수행하고 예측 결과를 저장하는 함수
    
    Args:
        model_type (str): 사용할 모델 유형 ('yolo', 'clip_ebc' 등)
        model_path (str, optional): 모델 파일 경로
        **kwargs: 모델별 추가 매개변수
    """
    # 모델 팩토리를 통해 모델 생성
    model = CrowdModelEval.get_model(model_type, model_path)
    
    os.makedirs(PRED_DIR, exist_ok=True)
    
    csv_files = glob.glob(os.path.join(LABEL_DIR, "*.csv"))
    print(f"{len(csv_files)}개의 CSV 파일을 찾았습니다.")
    print(f"사용 모델: {model_type}")
    
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
                # 어댑터를 통한 예측
                count = model.predict(image_path, **kwargs)
                df.at[idx, 'pred'] = count
                processed_count += 1
            except Exception as e:
                print(f"  이미지 처리 오류 ({image_name}): {e}")
        
        # 출력 파일명에 모델 유형 추가
        output_file = os.path.join(PRED_DIR, f"{model_type}_{os.path.basename(csv_file)}")
        df.to_csv(output_file, index=False)
        print(f"  처리 완료: {processed_count}/{len(df)} 이미지, 결과 저장됨: {output_file}")
    
    print("\n모든 이미지 처리 및 예측이 완료되었습니다.")


# 사용 예시
if __name__ == "__main__":
    # YOLO 모델 사용 예
    evaluate_images(model_type='yolo',model_path= PERSON_DET, conf=0.15, classes=[0])

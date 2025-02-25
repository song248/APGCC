import os
import glob
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math

def calculate_metrics(pred_dir='pred'):
    """
    Micro 및 Macro 관점에서 MSE, RMSE, MAE를 계산하는 함수
    
    Args:
        pred_dir (str): 예측 결과가 저장된 디렉토리 경로
    
    Returns:
        dict: Micro 및 Macro 관점의 평가 지표
    """
    # 모든 CSV 파일 찾기
    csv_files = glob.glob(os.path.join(pred_dir, "*.csv"))
    print(f"{len(csv_files)}개의 CSV 파일을 찾았습니다.")
    
    # Micro 평가를 위한 모든 예측과 실제값
    all_predictions = []
    all_actual = []
    
    # Macro 평가를 위한 파일별 지표
    file_metrics = []
    
    for csv_file in csv_files:
        file_name = os.path.basename(csv_file)
        try:
            # CSV 파일 로드
            df = pd.read_csv(csv_file)
            
            # NaN 값 제거
            df = df.dropna(subset=['pred'])
            
            if len(df) == 0:
                print(f"  {file_name}: 유효한 예측값이 없습니다.")
                continue
            
            # 실제값과 예측값 추출
            actual = df['crowd_counting'].values
            predictions = df['pred'].values
            
            # Micro 평가용 리스트에 추가
            all_predictions.extend(predictions)
            all_actual.extend(actual)
            
            # 각 파일별 지표 계산
            file_mse = mean_squared_error(actual, predictions)
            file_rmse = math.sqrt(file_mse)
            file_mae = mean_absolute_error(actual, predictions)
            
            # 각 파일별 지표 저장
            file_metrics.append({
                'file': file_name,
                'mse': file_mse,
                'rmse': file_rmse,
                'mae': file_mae,
                'samples': len(df)
            })
            
            print(f"  {file_name}: RMSE = {file_rmse:.2f}, MAE = {file_mae:.2f}, MSE = {file_mse:.2f}, 샘플 수 = {len(df)}")
            
        except Exception as e:
            print(f"  {file_name} 처리 오류: {e}")
    
    if not all_predictions:
        print("유효한 예측 결과가 없습니다.")
        return None
    
    # Micro 관점 평가 (모든 샘플을 동등하게 취급)
    micro_mse = mean_squared_error(all_actual, all_predictions)
    micro_rmse = math.sqrt(micro_mse)
    micro_mae = mean_absolute_error(all_actual, all_predictions)
    
    # Macro 관점 평가 (모든 파일을 동등하게 취급 - 단순 평균)
    macro_mse = np.mean([m['mse'] for m in file_metrics])
    macro_rmse = np.mean([m['rmse'] for m in file_metrics]) 
    macro_mae = np.mean([m['mae'] for m in file_metrics])
    
    # 가중 Macro 관점 평가 (파일별 샘플 수로 가중 평균)
    weights = [m['samples'] for m in file_metrics]
    weighted_macro_mse = np.average([m['mse'] for m in file_metrics], weights=weights)
    weighted_macro_rmse = np.average([m['rmse'] for m in file_metrics], weights=weights)
    weighted_macro_mae = np.average([m['mae'] for m in file_metrics], weights=weights)
    
    print("\n=== 평가 결과 요약 ===")
    print("Micro 관점 (개별 샘플 동등 취급):")
    print(f"  MSE = {micro_mse:.2f}, RMSE = {micro_rmse:.2f}, MAE = {micro_mae:.2f}")
    print("Macro 관점 (각 파일 동등 취급):")
    print(f"  MSE = {macro_mse:.2f}, RMSE = {macro_rmse:.2f}, MAE = {macro_mae:.2f}")
    print("가중 Macro 관점 (파일별 샘플 수로 가중치 부여):")
    print(f"  MSE = {weighted_macro_mse:.2f}, RMSE = {weighted_macro_rmse:.2f}, MAE = {weighted_macro_mae:.2f}")
    print(f"총 샘플 수: {len(all_predictions)}, 총 파일 수: {len(file_metrics)}")
    
    os.makedirs("metrics_result", exist_ok=True)
    # 결과를 CSV 파일로 저장
    results_df = pd.DataFrame(file_metrics)
    results_df.to_csv('metrics_result/file_evaluation_results.csv', index=False)
    
    # 요약 결과 저장
    summary_df = pd.DataFrame({
        'perspective': ['Micro', 'Macro', 'Weighted Macro'],
        'MSE': [micro_mse, macro_mse, weighted_macro_mse],
        'RMSE': [micro_rmse, macro_rmse, weighted_macro_rmse],
        'MAE': [micro_mae, macro_mae, weighted_macro_mae]
    })
    summary_df.to_csv('metrics_result/summary_evaluation_results.csv', index=False)
    
    print("평가 결과가 저장되었습니다.")
    
    return {
        'micro': {'mse': micro_mse, 'rmse': micro_rmse, 'mae': micro_mae},
        'macro': {'mse': macro_mse, 'rmse': macro_rmse, 'mae': macro_mae},
        'weighted_macro': {'mse': weighted_macro_mse, 'rmse': weighted_macro_rmse, 'mae': weighted_macro_mae},
        'total_samples': len(all_predictions),
        'total_files': len(file_metrics)
    }

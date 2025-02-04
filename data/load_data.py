import pickle
import glob
import os
from .process import get_audio_paths

def load_data(root_dir: str = None, test: bool = False):
    """
    오디오 데이터 경로와 label 을 불러오는 메서드.

    Args:
        root_dir (str): 데이터의 루트 경로
        test (bool): test 환경 여부

    Returns:
        List, List : 오디오 경로 리스트, label 리스트
    """
    try:
        if test:
            get_audio_paths(test=True)
            
        # 현재 파일의 위치를 기준으로 src 폴더 상대 경로를 생성
        src_folder = os.path.join(os.path.dirname(__file__), '..', 'src')
        data_dict = {}
        data_type_folder = os.path.join(src_folder, data_type.lower())
        if not os.path.isdir(data_type_folder):
            return {'status': 404, 
                    'info': f'Invalid data type. data_type : {data_type}'}
        pkl_paths = glob.glob(os.path.join(data_type_folder, '*.pkl'))
        for pkl in pkl_paths:
            filename = os.path.basename(pkl)
            filetext = os.path.splitext(filename)[0]
            with open(pkl, 'rb') as f:
                data = pickle.load(f)
            data_dict[filetext] = data

        return data_dict
    except Exception as e:
        print(e)
        return {'info': f'<Error message> : {e} '}

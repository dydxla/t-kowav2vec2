import pickle
import glob
import os

def load_data(data_type=None):
    try:
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

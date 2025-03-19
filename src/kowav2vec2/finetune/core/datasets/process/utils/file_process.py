import os

def path_to_basename(file_path: str) -> str:
    """
    파일의 절대 경로를 확장자를 제거한 파일이름으로 반환하는 매서드
    """
    base_name = os.path.basename(file_path)
    split_name = os.path.splitext(base_name)[0]
    return split_name
    
import os
from pathlib import Path
import glob
from typing import Dict, List
from kowav2vec2.finetune.core.datasets.process.utils import find_all_label_files

def get_label_paths(
        train_root_dir: str = None,
        eval_root_dir: str = None,
        format: str = "txt",
        test: bool = False
    ) -> Dict:
    """
    label 데이터들의 경로를 반환하는 메서드.

    Args:
        train_root_dir (str): 불러 오고자 하는 label 파일(train)들이 담긴 경로 (default: None)
        eval_root_dir (str): 불러 오고자 하는 label 파일(validation)들이 담긴 경로 (default: None)
        format (str): 불러 오고자 하는 label 파일의 확장자
        test (bool): test 환경 실행 여부 (default: False)
        
    Returns:
        Dict : 불러 오고자 하는 label txt 파일들의 경로들을 담은 딕셔너리
    """
    
    # test 환경인 경우
    # samples/test/label 폴더로 부터 label 파일 경로들을 가져옴
    if test:
        current_file = Path(__file__).resolve()
        txt_files = find_all_label_files(
            root_dir=current_file.parents[5] / "samples" / "label" / "test",
            format="txt"
        )
        return {'train': txt_files}
    
    # train_root_dir 이 None 인 경우
    # samples/train/label 폴더로 부터 label 파일 경로들을 가져옴
    if not train_root_dir:
        current_file = Path(__file__).resolve()
        txt_files = find_all_label_files(
            root_dir=current_file.parents[5] / "samples" / "label" / "train",
            format="txt"
        )
        # eval_root_dir 이 None 인 경우
        # samples/validation/label 폴더로 부터 label 파일 경로들을 가져옴
        if not eval_root_dir:
            val_txt_files = find_all_label_files(
                root_dir=current_file.parents[5] / "samples" / "label" / "validation",
                format="txt"
            )
            return {"train": txt_files, "test": val_txt_files}
        
        val_txt_files = find_all_label_files(
            root_dir=eval_root_dir,
            format=format
        )
        return {"train": txt_files, "test": val_txt_files}
    
    txt_files = find_all_label_files(
        root_dir=train_root_dir,
        format=format
    )
    
    if eval_root_dir:
        val_txt_files = find_all_label_files(
            root_dir=eval_root_dir,
            format=format
        )
        return {"train": txt_files, "test": val_txt_files}
    
    return {"train": txt_files}


def file_to_label(file_path):
    """
    txt file 로 부터 label 읽어 오는 매서드
    """
    with open(file_path, 'r', encoding="utf-8") as f:
        lines = f.readlines()
    return lines[0]
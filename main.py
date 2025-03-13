# source import
import train

# standard import
import argparse

if __name__=='__main__':
    parser = argparse.ArgumentParser(description="deepspeed 모델 훈련을 위한 인자값 전달")
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('-d', '--data_type', type=str, required=True)
    parser.add_argument('-t', '--tokenizer', type=str, required=True)
    parser.add_argument('-pm', '--pre', type=str, required=False)
    parser.add_argument('-cm', '--checkpoint', type=str, required=False)
    parser.add_argument('-o', '--output', type=str, required=True)
    parser.add_argument('-c', '--config', type=str, required=True)
    opt = parser.parse_args()
    data_type, tokenizer_dir, pre_model_dir, checkpoint_model_dir, output_dir, config_dir = opt.data_type, opt.tokenizer, opt.pre, opt.checkpoint, opt.output, opt.config
    
    if pre_model_dir is None and checkpoint_model_dir is None:
	    parser.error("Either -pm or -cm must be provided.")

    train.Train(data_type, tokenizer_dir, pre_model_dir, checkpoint_model_dir, output_dir, config_dir)

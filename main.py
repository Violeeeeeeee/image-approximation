import torch
from utils.config_loader import load_config, parse_args
from modules.evo_pipeline import EvoPipeline

def main():
    args = parse_args()

    print(f"Loading config from {args.config}...")
    config = load_config(args.config)

    # Enable cuDNN benchmarking to optimize performance for fixed input sizes
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    pipeline = EvoPipeline(config)
    pipeline.run()

if __name__ == '__main__':
    main()

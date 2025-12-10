import yaml
import argparse
from typing import Dict, Any

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Loads configuration settings from a YAML file.

    Args:
        config_path (str): The file path to the YAML configuration.

    Returns:
        Dict[str, Any]: A dictionary containing configuration parameters.

    Raises:
        SystemExit: If the file is not found or is invalid YAML.
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {config_path}")
        exit(1)
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file {config_path}: {e}")
        exit(1)

def parse_args() -> argparse.Namespace:
    """
    Parses command-line arguments.
    Primarily used to specify the path to the configuration file.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Run Evolutionary Image Approximation Experiment.")
    parser.add_argument(
        '--config',
        type=str,
        default='configs/base_config.yaml',
        help='Path to the YAML configuration file.'
    )
    return parser.parse_args()

if __name__ == '__main__':
    # Unit test block
    test_config_path = '../../configs/config.yaml'
    print(f"Testing config loading from {test_config_path}...")
    try:
        config = load_config(test_config_path)
        print("Successfully loaded config:")
        print(config)
    except Exception as e:
        print(f"Test failed: {e}")

import argparse

def parse_command_line() -> object:
    # Get commnad line arguments
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_dir', type=str, default='data/quixote', help='Model directory')
    parser.add_argument('--mode', type=str, default='character', help='Vocabulary mode: character or word')
    return parser.parse_args()

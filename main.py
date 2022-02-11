import argparse

from train import train

parser = argparse.ArgumentParser(description='IMDB model train.')
parser.add_argument('--hidden-dim', type=int, default=256)
parser.add_argument('--embed-dim', type=int, default=128)
parser.add_argument('--num-layers', type=int, default=1)
parser.add_argument('--num-classes', type=int, default=2)
parser.add_argument('--batch-size', type=int, default=64)
parser.add_argument('--training-epochs', type=int, default=10)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--model-type', type=str, default='gru')
parser.add_argument('--alpha', type=int, default=1)
parser.add_argument('--epsilon', type=int, default=0.02)
parser.add_argument('--save-path', type=str, default='results.csv')


args = parser.parse_args()


if __name__ == "__main__":
    train(args)

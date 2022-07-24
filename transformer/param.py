import argparse


def parse_args(gen=False):
    parser = argparse.ArgumentParser()
    args = parser.add_argument_group('Dataset options')
    args.add_argument('--batch_size_per_device', type=int, default=8)
    args.add_argument('--device', type=str, default='cpu')
    args.add_argument('--create_data', action='store_false')

    args = parser.parse_args()

    return args
from tttai import main as make_ai
from convert import export_weights
from cerate_dataset import main as dataset
import sys

SAMPLES = int(sys.argv[1])


def main():
    dataset(SAMPLES)
    make_ai(SAMPLES)
    export_weights(SAMPLES)


if __name__ == '__main__':
    main()

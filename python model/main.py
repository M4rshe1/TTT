from tttai import main as make_AI
from convert import export_weights
from cerate_dataset import main as Datadset
import sys

SAMPLES = int(sys.argv[1])


def main():
    Datadset(SAMPLES)
    make_AI(SAMPLES)
    export_weights(SAMPLES)


if __name__ == '__main__':
    main()

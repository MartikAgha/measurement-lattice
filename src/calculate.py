import os
import argparse
import string
from datetime import datetime

import numpy as np
import numpy.random as nr

from lattice_model.lattice_analysis import LatticeAnalysisRepeater

STR_LENGTH = 8

def get_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--length', type=int, default=10)
    parser.add_argument('-w', '--width', type=int, default=4)
    parser.add_argument('-np', '--number-probabilities', type=int, default=20)
    parser.add_argument('-nr', '--number-repeats', type=int, default=50)
    parser.add_argument('-p', '--parallelise', action='store_true')
    parser.add_argument('-min', '--probability-min', type=float, default=0.05)
    parser.add_argument('-max', '--probability-max', type=float, default=0.95)
    parser.add_argument('-c', '--calculation', type=str, default=None)
    parser.add_argument('-ac', '--all-calculations', action='store_true')
    parser.add_argument('-f', '--full-data', action='store_true')
    parser.add_argument('-d', '--directory', type=str, default='../data')
    parser.add_argument('-o', '--output-file', type=str, default=None)
    return parser

def main():
    parser = get_argparser()
    args = parser.parse_args()

    analyser = LatticeAnalysisRepeater(xdim=args.length,
                                       ydim=args.width,
                                       zdim=args.width,
                                       parallel=args.parallelise)

    probability_range = np.linspace(args.probability_min,
                                    args.probability_max,
                                    args.number_probabilities)

    all_calcs = analyser.possible_calculations()
    calcs = [args.calculation] if args.calculation is not None else all_calcs

    analyser.perform_analysis(calculations=calcs,
                              probability=probability_range,
                              n_repeats=args.number_repeats)

    data = analyser.get_data(agg=bool(not args.full_data))

    if args.output_file is None:
        out_str = ''.join(nr.choice(list(string.ascii_lowercase+
                                         string.ascii_uppercase), STR_LENGTH))
        date_part = datetime.strftime(datetime.now(), '%y_%m_%d')
        output_name = date_part + '_' + out_str + '.csv'
    else:
        output_name = args.output_file

    full_path = os.path.join(args.directory, output_name)
    data.to_csv(full_path)

if __name__ == '__main__':
    main()
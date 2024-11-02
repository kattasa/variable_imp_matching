from scate_coverage_exp import make_data
from argparse import ArgumentParser

def main():
    parser = ArgumentParser()
    parser.add_argument('--query_seed', type = int)
    parser.add_argument('--sample_seed', type = int)
    parser.add_argument('--n_train', type = int)
    parser.add_argument('--n_est', type = int)
    parser.add_argument('--n_query', type = int)
    parser.add_argument('--n_imp', type = int)
    parser.add_argument('--n_iter', type = int)
    parser.add_argument('--n_unimp', type = int)
    parser.add_argument('--dgp', type = str)
    parser_args = parser.parse_args()

    parser_args_dict = parser_args.__dict__

    make_data(**parser_args_dict)

if __name__ == '__main__':
    main()
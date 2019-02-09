import argparse
import pandas as pd
import sys


def generate_table(auprc_dict) :
    for i, d in enumerate(auprc_dict.keys()) :
        df = auprc_dict[d]
        if i == 0 :
            tstr = 'Data set '
            for m in df['method'] :
                tstr += ('& ' + m + ' ')
            print('-'* 30)
            print(tstr)
            print('-' * 30)
        pstr = d + ' \t'
        for m, amean, astd in zip(df['method'], df['auprc'], df['auprc_std']) :
            pstr += ('& %0.5f $\pm$ %0.5f ' % (amean, astd))
        print(pstr + ' \\\\')
                    
# -- end function

def generate_table2(auprc_dict, ntables=1) :
    nsets = len(auprc_dict.keys())
    start_idx = [ i * int(float(nsets) / float(ntables)) for i in range(ntables) ]
    start_idx.append(nsets)

    mlist = None
    for d in auprc_dict :
        mlist = auprc_dict[d]['method'].unique()

    for idx, sidx in enumerate(start_idx) :
        if sidx == nsets : break
        tstr = 'Method '
        for i, d in enumerate(sorted(auprc_dict.keys())) :
            if i < sidx : continue
            if i >= start_idx[idx + 1] : break
            tstr += ('& ' + d + ' ')
        print('\hline')
        print(tstr + '\\\\')
        print('\hline')

        for m in mlist :
            pstr = m + '\t'
            for i, d in enumerate(sorted(auprc_dict.keys())) :
                if i < sidx : continue
                if i >= start_idx[idx + 1] : break
                df = auprc_dict[d][auprc_dict[d]['method'] == m]
                for m1, amean, astd in zip(df['method'], df['auprc'], df['auprc_std']) :
                    assert m == m1
                    pstr += ('& %0.5f $\pm$ %0.5f ' % (amean, astd))
            print(pstr + ' \\\\')
        print('\hline')

# -- end function


def extract_data(fname) :
    print('Processing ', fname)
    df = pd.read_csv(fname)
    return df[df['method'] != 'RPTree+']
# -- end function


def main() :
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-r',
        '--results_files',
        help='Comma-separated list of files containing the PR-curve',
        type=str
    )
    parser.add_argument(
        '-d',
        '--data_names',
        help='Comma-separated list of data set names corresponding to the data files',
        type=str
    )
    args = parser.parse_args()

    auprc_dfs = [
        extract_data(fname)
        for fname in args.results_files.split(',')
    ]
    dnames = [ n for n in args.data_names.split(',') ]
    assert len(auprc_dfs) == len(dnames)
    
    auprc_dict = { n : df for n, df in zip(dnames, auprc_dfs) }

    generate_table(auprc_dict)
    generate_table2(auprc_dict, 2)
# -- end function


if __name__ == '__main__' :
    status = main()
    sys.exit(status)
# -- end function

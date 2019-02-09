import argparse
import pandas as pd
import sys

from matplotlib import pyplot as plt

SMALL_SIZE = 10
MEDIUM_SIZE = 16
BIGGER_SIZE = 16

#plt.rc('font', size=BIGGER_SIZE)         # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)    # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


def generate_figures(prc_dict, figfile) :

    my_colors = [
        'tab:blue',
        'tab:olive',
        'tab:orange',
        'tab:green',
        'tab:red',
        'tab:purple',
        'tab:brown',
        'tab:pink',
        'tab:gray'
    ]
    my_markers = 'ov^s+*xdp'
    color_dict = {}
    marker_dict = {}

    nsets = len(prc_dict.keys())

    # extract list of method
    mlist = None
    for dname in prc_dict :
        mlist = prc_dict[dname]['method'].unique()
        break
    
    for method, col, mrk in zip(mlist, my_colors, my_markers) :
        color_dict[method] = col
        marker_dict[method] = mrk

    print(color_dict)
    print(marker_dict)

    base = 3
    nrows = 2
    ncols = nsets / nrows
    plt.figure(1, figsize=(ncols  * base, nrows * base))
    #   a. Plot PR-curve for each data (line plots) 
    for idx, dname in enumerate(sorted(prc_dict.keys())) :
        print('Processing %s (%i/%i) ...' % (dname, idx + 1, nsets))
        df = prc_dict[dname]
        
        #   a. Plot PR-curve for each method (line plots) 
        grouped_pr_curves = df.groupby(['method'])
        plt.subplot(nrows, ncols, idx + 1)
        for method, values in grouped_pr_curves :
            plt.plot(
                values['recall'], 
                values['precision'],
                label=method,
                color=color_dict[method],
                marker=marker_dict[method],
                markersize=6
            )
            if idx % ncols == 0 :
                plt.ylabel('Precision')
            plt.xlabel('Recall')
            plt.title(dname, fontsize=BIGGER_SIZE)

    plt.legend(
        loc='upper right',
        framealpha=1.0,
        shadow=True,
        bbox_to_anchor=(0.7,-0.3),
        borderaxespad=0.,
        ncol=4
    )

    plt.tight_layout()
    print('Plots generated, saving figures in \'%s\'' % figfile)
    plt.savefig(fname=figfile, format='png')    
# -- end function


def extract_data(fname) :
    print('Processing ', fname)
    df = pd.read_csv(fname)[ [ 'method', 'precision', 'recall' ] ]
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
    parser.add_argument(
        '-g',
        '--figures_file',
        help='File where the figures will be output for this experiment',
        type=str
    )
    args = parser.parse_args()

    prc_dfs = [
        extract_data(fname)
        for fname in args.results_files.split(',')
    ]
    dnames = [ n for n in args.data_names.split(',') ]
    assert len(prc_dfs) == len(dnames)
    
    prc_dict = { n : df for n, df in zip(dnames, prc_dfs) }

    generate_figures(prc_dict, args.figures_file)
# -- end function


if __name__ == '__main__' :
    status = main()
    sys.exit(status)
# -- end function

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
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
#plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title



def generate_figures(all_results, figfile) :

    my_colors = [
        'tab:red',
        'tab:gray',
        'tab:blue',
        'tab:green',
        # 'tab:olive',
        # 'tab:orange',
        # 'tab:purple',
        # 'tab:brown',
        # 'tab:pink',
    ]
    my_markers = 'o+*xd' + 'v^s+' # + '*xdp'
    my_styles = [ 'solid', 'dashed', 'dashdot' ]

    mlist = all_results['method'].unique()
    color_dict = { m : c for m, c in zip(mlist, my_colors) }
    style_dict = { m : s for m, s in zip(mlist, my_styles) }

    print(color_dict)
    
    dlist = all_results['d'].unique()
    marker_dict = { d : m for d, m in zip(dlist, my_markers) }

    print(marker_dict)

    grouped_curves = all_results.groupby(['method', 'd'])

    plt.figure(1, figsize=(7,5))
    for (method, dim), values in grouped_curves :
        plt.plot(
            values['n'], 
            values['mean_query_time'],
            label=(method + '(d=' + str(dim) + ')'),
            color=color_dict[method],
            # ls=style_dict[method],
            marker=marker_dict[dim],
            markersize=7,
        )
    plt.legend(
        loc='lower left',
        ncol=3,
        framealpha=0.1,
        bbox_to_anchor=(0,1),
        borderaxespad=0.
    )
    plt.xscale('log', basex=2)
    plt.yscale('log', basey=2)
    plt.xlabel('Number of points')
    plt.ylabel(r'Query time ratio wrt $d = 2^7, n = 2^{10}$')
    #plt.title(r'Scaling with $d$ and $n$', fontsize=BIGGER_SIZE)

    plt.tight_layout()
    print('Plots generated, saving figures in \'%s\'' % figfile)
    plt.savefig(fname=figfile, format='png')    
# -- end function

def prepare_scaling_data(res_files_list, dims_list) :
    flist = res_files_list.split(',')
    dlist = [ int(d) for d in dims_list.split(',') ]
    assert len(flist) == len(dlist)
    fdict = {}
    for d,f in zip(dlist, flist) :
        print('%i --> %s' % (d, f))
        fdict[d] = f

    # Reading in the data 
    min_d = min(dlist)
    scale_base = None
    scaling_dict = {}
    for d in sorted(fdict.keys()) :
        f = fdict[d]
        print('Reading in %s for d=%i' % (f, d))
        d_res = pd.read_csv(f)
        d_res['d'] = [ d ] * len(d_res)
        # print(d_res.head())
        scaling_dict[d] = d_res
        if d == min_d :
            min_n = d_res['n'].min()
            print('Minimum n=%i' % min_n)
            scale_base = d_res[d_res['n'] == min_n]

    # Scale all the data frames
    print('The base for scaling')
    print(scale_base.head(10))
    methods = scale_base['method'].unique()
    assert len(methods) == len(scale_base)

    mean_col_list = [ 'mean_index_size', 'mean_indexing_time', 'mean_query_time' ]
    std_col_list = [ 'std_index_size', 'std_indexing_time', 'std_query_time' ]

    for d in sorted(scaling_dict.keys()) :
        #print('d =', d)
        df = scaling_dict[d]
        for m in methods :
            #print('m =', m)
            for cm, cs in zip(mean_col_list, std_col_list) :
                base = scale_base[scale_base['method'] == m][cm].values[0]
                #print('Base %s = %g' %(cm, base))
                df.loc[df['method'] == m, cm] /= base
                df.loc[df['method'] == m, cs] /= base

        #print(df.head(3))
        #print(df.tail(3))

    # Merge all the data frames
    all_scaling_res = pd.concat([ scaling_dict[d] for d in sorted(scaling_dict.keys()) ])
    print 'Final results table', all_scaling_res.shape
    print(all_scaling_res.head(6))
    print(all_scaling_res.tail(6))

    return all_scaling_res
# -- end function


def main() :
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-r',
        '--results_files',
        help='Comma-separated list of files containing the scaling data',
        type=str
    )
    parser.add_argument(
        '-d',
        '--dimensions',
        help='Comma-separated list of dimensions corresponding to the data files',
        type=str
    )
    parser.add_argument(
        '-g',
        '--figures_file',
        help='File where the figures will be output for this experiment',
        type=str
    )
    args = parser.parse_args()

    results_df = prepare_scaling_data(args.results_files, args.dimensions)
    generate_figures(results_df, args.figures_file)

    return 0
# -- end functions

if __name__ == '__main__' :
    status = main()
    sys.exit(status)
# -- end function

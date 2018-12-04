from matplotlib import pyplot as plt

def generate_figures(all_results, all_auprcs, figfile) :
    my_colors = 'rgbkymc'
    my_markers = 'o^s+*xd'
    color_dict = {}
    marker_dict = {}
    for method, col, mrk in zip(all_auprcs['method'], my_colors, my_markers) :
        color_dict[method] = col
        marker_dict[method] = mrk

    grouped_pr_curves = all_results.groupby(['method'])

    plt.figure(1, figsize=(5,12))
    #   a. Plot PR-curve for each method (line plots) 
    plt.subplot(311)
    for method, values in grouped_pr_curves :
        plt.plot(
            values['recall'], 
            values['precision'],
            label=method,
            color=color_dict[method],
            marker=marker_dict[method]
        )
    plt.legend(loc='lower left', framealpha=0.5, fontsize=7)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Recall vs. Precision')

    #   b. Plot AUPRC for each method (bar plots)
    plt.subplot(312)
    color_list = [ color_dict[m] for m in all_auprcs['method'] ]
    plt.bar(
        all_auprcs['method'],
        all_auprcs['auprc'],
        color=color_list
    )
    plt.xticks(rotation=20)
    plt.xlabel('Method')
    plt.ylabel('AUPRC')
    plt.title('Area under Recall vs. Precision Curve')

    #   c. #queries/second curve for each method (line plots)
    plt.subplot(313)
    for method, values in grouped_pr_curves :
        plt.plot(
            values['recall'], 
            values['nqueries_per_sec'],
            label=method,
            color=color_dict[method],
            marker=marker_dict[method]
        )
    plt.yscale('log')
    plt.xlabel('Recall')
    plt.ylabel('#queries/sec')
    plt.title('Recall vs. Num. queries per second')

    plt.tight_layout()
    print('Plots generated, saving figures in \'%s\'' % figfile)
    plt.savefig(fname=figfile, format='png')    
# -- end function

import roman
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker
sns.set_palette("Set1")
sns.set_style("white")
plt.rcParams['font.sans-serif'] = 'Times New Roman'
plt.rcParams["mathtext.fontset"] = 'stix'
def paintzhuzhuang():
    # import matplotlib.pyplot as matplotlib.pyplot
    import matplotlib
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    # palette = matplotlib.pyplot.get_cmap('Set3')
    name_list = ['$\\it{w/o}$ Intra', '$\\it{w/o}$ Inter', '$\\it{w/o}$ Fusion', 'COFRAUD']
    num_list = [88.02, 92.9, 84.18, 97.21]
    num_list1 = [88.4, 88.36, 88.96, 89.08]
    num_list2 = [90.62, 91.25, 90.73, 91.53]
    x = np.arange(4)
    total_width, n = 0.6, 3
    width = total_width / n
    x = x - (total_width - width) / 2

    matplotlib.pyplot.figure(figsize=(7.1, 6))
    matplotlib.pyplot.subplots_adjust(left=0.2, bottom=0.2)
    matplotlib.pyplot.bar(x, num_list, width=width, label='AUC', tick_label=name_list, color="#4472C4", align='edge', lw=0.5)

    matplotlib.pyplot.bar(x + width, num_list1, width=width, label='Rec', tick_label=name_list, color="#ED7D31", align='edge', lw=0.5)

    matplotlib.pyplot.bar(x + 2 * width, num_list2, width=width, label='F1', tick_label=name_list, color="#A5A5A5", align='edge',
            lw=0.5)
    matplotlib.pyplot.grid(alpha=0.8, axis = 'y')
    matplotlib.pyplot.legend(loc = 'upper center', ncol = 3, fontsize = 24, columnspacing = 0.4, handletextpad = 0.2, markerscale = 0.5)
    matplotlib.pyplot.ylabel('Percentage', fontsize=28)
    matplotlib.pyplot.xticks(fontsize=28)
    matplotlib.pyplot.yticks(fontsize=28)
    matplotlib.pyplot.xticks([index + (total_width/2) for index in x], name_list, rotation = 25)
    matplotlib.pyplot.yticks(np.arange(80, 100.01, 5))
    matplotlib.pyplot.ylim(80,102)
    matplotlib.pyplot.savefig('amazonwo.pdf')
    matplotlib.pyplot.show()


    num_list = [91.17, 90.41, 87.19, 91.52]
    num_list1 = [72.09, 75.07, 67.58, 79.7]
    num_list2 = [75.51, 77, 70.7, 79.71]
    matplotlib.pyplot.figure(figsize=(7.1, 6))
    matplotlib.pyplot.subplots_adjust(left=0.2, bottom=0.2)
    matplotlib.pyplot.bar(x, num_list, width=width, label='AUC', tick_label=name_list , color = "#4472C4", align='edge', lw = 0.5 )

    matplotlib.pyplot.bar(x+ width, num_list1, width=width, label='Rec', tick_label=name_list,color = "#ED7D31", align='edge', lw = 0.5)

    matplotlib.pyplot.bar(x + 2 * width, num_list2, width=width, label='F1', tick_label=name_list,color = "#A5A5A5", align='edge', lw = 0.5)

    matplotlib.pyplot.grid(alpha=0.8, axis = 'y')
    matplotlib.pyplot.legend(loc = 'upper center', ncol = 3, fontsize = 24, columnspacing = 0.4, handletextpad = 0.2, markerscale = 0.5)
    matplotlib.pyplot.ylabel('Percentage', fontsize=28)
    matplotlib.pyplot.xticks(fontsize=28)
    matplotlib.pyplot.yticks([60, 70, 80, 90, 100], fontsize = 28)
    matplotlib.pyplot.xticks([index + (total_width/2) for index in x], name_list, rotation = 25)
    matplotlib.pyplot.ylim(60,100)
    matplotlib.pyplot.savefig('yelpwo.pdf')
    matplotlib.pyplot.show()

def f1():
    # import matplotlib
    # import matplotlib.pyplot as plt
    # import numpy as np
    # import matplotlib.ticker as ticker
    # import seaborn as sns
    # palette = matplotlib.pyplot.get_cmap('Set3')
    # Example data
    categories = []
    categories.append(['0.16%', '0.32%', '100%'])
    categories.append(['0.10%', '0.50%', '100%'])
    categories.append(['0.10%', '0.50%', '100%'])
    labels = ['Random', 'Degree', 'Herding', 'K-Center', 'VNG', 'MCond', 'Whole']
    data_category1 = []
    data_category2 = []
    data_category3 = []
    data_category4 = []
    data_category5 = []
    data_category6 = []
    data_category7 = []
    titles = []
    #Pubmed
    data_category1.append(np.array([1.98, 2.04]))
    data_category2.append(np.array([1.99, 2.05]))
    data_category3.append(np.array([1.98, 2.05]))
    data_category4.append(np.array([1.98, 2.04]))
    data_category5.append(np.array([2.48, 2.61]))
    data_category6.append(np.array([2.55, 3.88]))
    data_category7.append(np.array([38.62]))
    titles.append('(d) Pubmed')

    data_category1.append(np.array([2.01, 2.36]))
    data_category2.append(np.array([2.02, 2.39]))
    data_category3.append(np.array([2.01, 2.36]))
    data_category4.append(np.array([2.01, 2.36]))
    data_category5.append(np.array([3.04, 4.26]))
    data_category6.append(np.array([5.01, 9.20]))
    data_category7.append(np.array([133.30]))
    titles.append('(e) Flickr')

    data_category1.append(np.array([2.67, 4.11]))
    data_category2.append(np.array([2.76, 5.12]))
    data_category3.append(np.array([2.67, 4.11]))
    data_category4.append(np.array([2.67, 4.10]))
    data_category5.append(np.array([7.33, 19.98]))
    data_category6.append(np.array([10.12, 12.12]))
    data_category7.append(np.array([565.94]))
    titles.append('(f) Reddit')

    # data_category_log1 = [np.log(arr) for arr in data_category1]
    # data_category_log2 = [np.log(arr) for arr in data_category2]
    # data_category_log3 = [np.log(arr) for arr in data_category3]
    # data_category_log4 = [np.log(arr) for arr in data_category4]
    # data_category_log5 = [np.log(arr) for arr in data_category5]
    # data_category_log6 = [np.log(arr) for arr in data_category6]
    # data_category_log7 = [np.log(arr) for arr in data_category7]
    # Set the width of the bars
    bar_width = 0.153
    coloumn_width = 1

    # Set the positions of the bars on the x-axis
    r1 = np.arange(len(data_category1[0])) * coloumn_width
    print(r1)
    r2 = [x + bar_width for x in r1]
    r3 = [x + bar_width for x in r2]
    r4 = [x + bar_width for x in r3]
    r5 = [x + bar_width for x in r4]
    r6 = [x + bar_width for x in r5]
    r7 = [2 * r1[-1] - r1[-2]]
    fig, axes = plt.subplots(nrows = 1, ncols = 3, figsize=(22,5))
    # plt.style.use('fivethirtyeight')
    i = 0

    plt.rcParams['font.sans-serif'] ='Times New Roman'
    plt.rcParams["mathtext.fontset"]='stix'
    # colors = ["#4E79A7", "#A0CBE8", "#F28E2B", "#FFBE7D", "#59A14F", "#8CD17D", "#B6992D", "#F1CE63", "#499894",
    #           "#86BCB6", "#E15759", "#E19D9A"]
    # Create the bar chart
    for ax in fig.axes:
        ax.bar(r1, data_category1[i], edgecolor = 'black', linewidth = 0.8,width=bar_width, label=labels[0], zorder=10)
        ax.bar(r2, data_category3[i], edgecolor = 'black', linewidth = 0.8,width=bar_width, label=labels[1], zorder=10)
        ax.bar(r3, data_category3[i], edgecolor = 'black', linewidth = 0.8,width=bar_width, label=labels[2], zorder=10)
        ax.bar(r4, data_category4[i], edgecolor = 'black', linewidth = 0.8,width=bar_width, label=labels[3], zorder=10)
        ax.bar(r5, data_category5[i], edgecolor = 'black', linewidth = 0.8,width=bar_width, label=labels[4], zorder=10)
        ax.bar(r6, data_category6[i], edgecolor = 'black', linewidth = 0.8,width=bar_width, label=labels[5], zorder=10)
        ax.bar(r7, data_category7[i], edgecolor = 'black', linewidth = 0.8,width=bar_width, label=labels[6], zorder=10)
    # Set the x-axis labels and title
        ax.set_xticks([0.36, 1.36, 2])
        ax.tick_params(bottom=False, top=False, left=False, right=False)
        ax.set_yticklabels(ax.get_yticks(), fontsize=16)
        ax.set_xticklabels(categories[i], fontsize=16)
        ax.set_xlabel('Reduction ratio', fontsize=18)
        ax.set_ylabel('Memory $(MB)$', fontsize=18)
        ax.set_title(titles[i], fontsize = 20, y = -0.30)
        ax.grid(True, alpha=0.5,  axis='y',color='gray', linewidth=1.5, linestyle='--')
        # ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

        fontsize1 = 12
        for ii, data in enumerate(data_category1[i]):
            ax.text(ii, data, str(data), ha='center', va='bottom', zorder = 11, fontsize = fontsize1)
        for ii, data in enumerate(data_category2[i]):
            ax.text(ii+bar_width*1, data, str(data), ha='center', va='bottom', zorder = 11, fontsize = fontsize1)
        for ii, data in enumerate(data_category3[i]):
            ax.text(ii+bar_width*2, data, str(data), ha='center', va='bottom', zorder = 11, fontsize = fontsize1)
        for ii, data in enumerate(data_category4[i]):
            ax.text(ii+bar_width*3, data, str(data), ha='center', va='bottom', zorder = 11, fontsize = fontsize1)
        for ii, data in enumerate(data_category5[i]):
            ax.text(ii+bar_width*4, data, str(data), ha='center', va='bottom', zorder = 11, fontsize = fontsize1)
        for ii, data in enumerate(data_category6[i]):
            ax.text(ii+bar_width*5, data, str(data), ha='center', va='bottom', zorder = 11, fontsize = fontsize1)
        for ii, data in enumerate(data_category7[i]):
            ax.text(ii+2, data, str(data), ha='center', va='bottom', zorder = 11, fontsize = fontsize1)


        arrow_length = 0.6  # Proportion of arrow length to the total distance
        arrow_params1 = {'arrowstyle': ']-', 'linewidth': 2.5, 'color': 'black', 'alpha': 0.3}
        arrow_params2 = {'arrowstyle': '->', 'linewidth': 2.5,'color': 'black', 'alpha': 0.3}
        for j in range(len(data_category1[i])):
            # ax.annotate('', xytext=(r1[j], max(data_category7[i])), xy=(r1[j], data_category1[i][j] + 0.05 * (max(data_category7[i])- data_category1[i][j])), arrowprops=arrow_params, annotation_clip=False)
            # ax.annotate('', xytext=(r2[j], max(data_category7[i])), xy=(r2[j], data_category2[i][j] + 0.05 * (max(data_category7[i])- data_category2[i][j])), arrowprops=arrow_params, annotation_clip=False)
            # ax.annotate('', xytext=(r3[j], max(data_category7[i])), xy=(r3[j], data_category3[i][j] + 0.05 * (max(data_category7[i])- data_category3[i][j])), arrowprops = arrow_params, annotation_clip = False)
            # ax.annotate('', xytext=(r4[j], max(data_category7[i])), xy=(r4[j], data_category4[i][j] + 0.05 * (max(data_category7[i])- data_category4[i][j])), arrowprops = arrow_params, annotation_clip = False)
            # ax.annotate('', xytext=(r5[j], max(data_category7[i])), xy=(r5[j], data_category5[i][j] + 0.05 * (max(data_category7[i])- data_category5[i][j])), arrowprops = arrow_params, annotation_clip = False)
            ax.annotate("" ,
                        xytext=(r6[j], max(data_category7[i]) * 0.99),
                        xy=(r6[j], data_category6[i][j] + 0.64 * (max(data_category7[i]) - data_category6[i][j])),
                        arrowprops=arrow_params1, annotation_clip=False)
            ax.annotate("%.1fx\nsmaller" % max(data_category7[i]/data_category6[i][j]), xytext=(r6[j]-0.120, data_category6[i][j] + 0.5 * (max(data_category7[i]) - data_category6[i][j])), xy=(r6[j], data_category6[i][j] + 0.04 * (max(data_category7[i])- data_category6[i][j])), arrowprops = arrow_params2, annotation_clip = False, fontsize = 15)
        # if i == 0:
        #     ax.set_ylim(0, 35)
        i = i + 1

    lines, labels = fig.axes[-1].get_legend_handles_labels()
    # Add a legend
    plt.subplots_adjust(top = 0.860, bottom = 0.2000, left = 0.050, right = 0.990, wspace=0.230, hspace=0.200)

    fig.legend(lines, labels, loc = 'upper center', fontsize = 18, ncol = 7)
    # plt.tight_layout()
    # Show the plot
    plt.savefig('node_memory.pdf', bbox_inches='tight')
    # plt.show()
def f2():

    # palette = matplotlib.pyplot.get_cmap('Set3')
    # Example data
    categories = []
    categories.append(['0.16%', '0.32%', '100%'])
    categories.append(['0.10%', '0.50%', '100%'])
    categories.append(['0.10%', '0.50%', '100%'])
    labels = ['Random', 'Degree', 'Herding', 'K-Center', 'VNG', 'MCond', 'Whole']
    data_category1 = []
    data_category2 = []
    data_category3 = []
    data_category4 = []
    data_category5 = []
    data_category6 = []
    data_category7 = []
    titles = []
    #Pubmed
    data_category1.append(np.array([4.17, 4.83]))
    data_category2.append(np.array([4.45, 4.95]))
    data_category3.append(np.array([3.67, 4.02]))
    data_category4.append(np.array([3.63, 4.17]))
    data_category5.append(np.array([7.17, 9.20]))
    data_category6.append(np.array([5.80, 6.96]))
    data_category7.append(np.array([135.85]))
    titles.append("(a) Pubmed")

    data_category1.append(np.array([5.39, 7.38]))
    data_category2.append(np.array([5.44, 7.82]))
    data_category3.append(np.array([5.12, 7.35]))
    data_category4.append(np.array([5.16, 7.34]))
    data_category5.append(np.array([11.18, 24.98]))
    data_category6.append(np.array([12.22, 30.83]))
    data_category7.append(np.array([631.03]))
    titles.append("(b) Flickr")

    data_category1.append(np.array([51.72, 53.62]))
    data_category2.append(np.array([51.04, 59.37]))
    data_category3.append(np.array([48.90, 50.88]))
    data_category4.append(np.array([48.70, 51.55]))
    data_category5.append(np.array([194.25, 1160.80]))
    data_category6.append(np.array([144.73, 416.67]))
    data_category7.append(np.array([17303.18 ]))
    titles.append("(c) Reddit")

    # Set the width of the bars
    # data_category_log1 = [np.log(arr) for arr in data_category1]
    # data_category_log2 = [np.log(arr) for arr in data_category2]
    # data_category_log3 = [np.log(arr) for arr in data_category3]
    # data_category_log4 = [np.log(arr) for arr in data_category4]
    # data_category_log5 = [np.log(arr) for arr in data_category5]
    # data_category_log6 = [np.log(arr) for arr in data_category6]
    # data_category_log7 = [np.log(arr) for arr in data_category7]
    # Set the width of the bars
    bar_width = 0.153
    coloumn_width = 1

    # Set the positions of the bars on the x-axis
    r1 = np.arange(len(data_category1[0])) * coloumn_width
    print(r1)
    r2 = [x + bar_width for x in r1]
    r3 = [x + bar_width for x in r2]
    r4 = [x + bar_width for x in r3]
    r5 = [x + bar_width for x in r4]
    r6 = [x + bar_width for x in r5]
    r7 = [2 * r1[-1] - r1[-2]]
    fig, axes = plt.subplots(nrows = 1, ncols = 3, figsize=(22,5))
    # plt.style.use('fivethirtyeight')
    i = 0
    # sns.set_palette("Set1")
    # sns.set_style("white")
    plt.rcParams['font.sans-serif'] ='Times New Roman'
    plt.rcParams["mathtext.fontset"]='stix'
    # colors = ["#4E79A7", "#A0CBE8", "#F28E2B", "#FFBE7D", "#59A14F", "#8CD17D", "#B6992D", "#F1CE63", "#499894",
    #           "#86BCB6", "#E15759", "#E19D9A"]
    # Create the bar chart
    for ax in fig.axes:

        ax.bar(r1, data_category1[i], edgecolor = 'black', linewidth = 0.8,width=bar_width, label=labels[0], zorder=10)
        ax.bar(r2, data_category3[i], edgecolor = 'black', linewidth = 0.8,width=bar_width, label=labels[1], zorder=10)
        ax.bar(r3, data_category3[i], edgecolor = 'black', linewidth = 0.8,width=bar_width, label=labels[2], zorder=10)
        ax.bar(r4, data_category4[i], edgecolor = 'black', linewidth = 0.8,width=bar_width, label=labels[3], zorder=10)
        ax.bar(r5, data_category5[i], edgecolor = 'black', linewidth = 0.8,width=bar_width, label=labels[4], zorder=10)
        ax.bar(r6, data_category6[i], edgecolor = 'black', linewidth = 0.8,width=bar_width, label=labels[5], zorder=10)
        ax.bar(r7, data_category7[i], edgecolor = 'black', linewidth = 0.8,width=bar_width, label=labels[6], zorder=10)
    # Set the x-axis labels and title
        ax.set_xticks([0.36, 1.36, 2])
        ax.tick_params(bottom=False, top=False, left=False, right=False)
        ax.set_yticklabels(ax.get_yticks(), fontsize=16)
        ax.set_xticklabels(categories[i], fontsize=16)
        ax.set_xlabel('Reduction ratio', fontsize=18)
        ax.set_ylabel('Time $(ms)$', fontsize=18)
        ax.set_title(titles[i], fontsize = 20, y = -0.30)
        ax.grid(True, alpha=0.5,  axis='y',color='gray', linewidth=1.5, linestyle='--')
        # ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        fontsize1 = 10
        for ii, data in enumerate(data_category1[i]):
            ax.text(ii, data, str(data), ha='center', va='bottom', zorder = 11, fontsize = fontsize1)
        for ii, data in enumerate(data_category2[i]):
            ax.text(ii+bar_width*1, data, str(data), ha='center', va='bottom', zorder = 11, fontsize = fontsize1)
        for ii, data in enumerate(data_category3[i]):
            ax.text(ii+bar_width*2, data, str(data), ha='center', va='bottom', zorder = 11, fontsize = fontsize1)
        for ii, data in enumerate(data_category4[i]):
            ax.text(ii+bar_width*3, data, str(data), ha='center', va='bottom', zorder = 11, fontsize = fontsize1)
        for ii, data in enumerate(data_category5[i]):
            ax.text(ii+bar_width*4, data, str(data), ha='center', va='bottom', zorder = 11, fontsize = fontsize1)
        for ii, data in enumerate(data_category6[i]):
            ax.text(ii+bar_width*5, data, str(data), ha='center', va='bottom', zorder = 11, fontsize =fontsize1)
        for ii, data in enumerate(data_category7[i]):
            ax.text(ii+2, data, str(data), ha='center', va='bottom', zorder = 11, fontsize = fontsize1)


        arrow_length = 0.6  # Proportion of arrow length to the total distance
        arrow_params1 = {'arrowstyle': ']-', 'linewidth': 2.5, 'color': 'black', 'alpha': 0.3}
        arrow_params2 = {'arrowstyle': '->', 'linewidth': 2.5,'color': 'black', 'alpha': 0.3}
        for j in range(len(data_category1[i])):
            # ax.annotate('', xytext=(r1[j], max(data_category7[i])), xy=(r1[j], data_category1[i][j] + 0.05 * (max(data_category7[i])- data_category1[i][j])), arrowprops=arrow_params, annotation_clip=False)
            # ax.annotate('', xytext=(r2[j], max(data_category7[i])), xy=(r2[j], data_category2[i][j] + 0.05 * (max(data_category7[i])- data_category2[i][j])), arrowprops=arrow_params, annotation_clip=False)
            # ax.annotate('', xytext=(r3[j], max(data_category7[i])), xy=(r3[j], data_category3[i][j] + 0.05 * (max(data_category7[i])- data_category3[i][j])), arrowprops = arrow_params, annotation_clip = False)
            # ax.annotate('', xytext=(r4[j], max(data_category7[i])), xy=(r4[j], data_category4[i][j] + 0.05 * (max(data_category7[i])- data_category4[i][j])), arrowprops = arrow_params, annotation_clip = False)
            # ax.annotate('', xytext=(r5[j], max(data_category7[i])), xy=(r5[j], data_category5[i][j] + 0.05 * (max(data_category7[i])- data_category5[i][j])), arrowprops = arrow_params, annotation_clip = False)
            ax.annotate("" ,
                        xytext=(r6[j] , max(data_category7[i]) * 0.99),
                        xy=(r6[j], data_category6[i][j] + 0.64 * (max(data_category7[i]) - data_category6[i][j])),
                        arrowprops=arrow_params1, annotation_clip=False)
            if max(data_category7[i]/data_category6[i][j]) > 100:
                ax.annotate("%.1fx\nfaster" % max(data_category7[i] / data_category6[i][j]), xytext=(
                r6[j] - 0.112, data_category6[i][j] + 0.5 * (max(data_category7[i]) - data_category6[i][j])),
                            xy=(r6[j], data_category6[i][j] + 0.04 * (max(data_category7[i]) - data_category6[i][j])),
                            arrowprops=arrow_params2, annotation_clip=False, fontsize=15)
            else:
                ax.annotate("%.1fx\nfaster" % max(data_category7[i]/data_category6[i][j]), xytext=(r6[j]-0.092, data_category6[i][j] + 0.5 * (max(data_category7[i]) - data_category6[i][j])), xy=(r6[j], data_category6[i][j] + 0.04 * (max(data_category7[i])- data_category6[i][j])), arrowprops = arrow_params2, annotation_clip = False, fontsize = 15)
        # if i == 0:
        #     ax.set_ylim(0, 35)
        i = i + 1

    lines, labels = fig.axes[-1].get_legend_handles_labels()
    # Add a legend
    plt.subplots_adjust(top = 0.860, bottom = 0.2000, left = 0.050, right = 0.990, wspace=0.230, hspace=0.200)

    fig.legend(lines, labels, loc = 'upper center', fontsize = 18, ncol = 7)
    # plt.tight_layout()
    # Show the plot
    plt.savefig('node_time.pdf', bbox_inches='tight')
    # plt.show()

def f3():
    # import matplotlib
    # import matplotlib.pyplot as plt
    # import numpy as np
    # import matplotlib.ticker as ticker
    # import seaborn as sns
    # palette = matplotlib.pyplot.get_cmap('Set3')
    # Example data
    categories = []
    categories.append(['0.16%', '0.32%', '100%'])
    categories.append(['0.10%', '0.50%', '100%'])
    categories.append(['0.10%', '0.50%', '100%'])

    labels = ['Random', 'Degree', 'Herding', 'K-Center', 'VNG', 'MCond', 'Whole']
    data_category1 = []
    data_category2 = []
    data_category3 = []
    data_category4 = []
    data_category5 = []
    data_category6 = []
    data_category7 = []
    titles = []
    #Pubmed
    data_category1.append(np.array([1.99, 2.05]))
    data_category2.append(np.array([1.99, 2.06]))
    data_category3.append(np.array([1.99, 2.05]))
    data_category4.append(np.array([1.99, 2.05]))
    data_category5.append(np.array([2.48, 2.61]))
    data_category6.append(np.array([2.56, 3.89]))
    data_category7.append(np.array([38.63]))
    titles.append('(d) Pubmed')

    data_category1.append(np.array([2.01, 2.36]))
    data_category2.append(np.array([2.03, 2.39]))
    data_category3.append(np.array([2.01, 2.36]))
    data_category4.append(np.array([2.01, 2.36]))
    data_category5.append(np.array([3.05, 4.26]))
    data_category6.append(np.array([5.97, 10.68]))
    data_category7.append(np.array([133.31]))
    titles.append('(e) Flickr')

    data_category1.append(np.array([2.68, 4.12]))
    data_category2.append(np.array([2.77, 5.14]))
    data_category3.append(np.array([2.68, 4.11]))
    data_category4.append(np.array([2.68, 4.11]))
    data_category5.append(np.array([7.33, 19.99]))
    data_category6.append(np.array([11.8, 13.9]))
    data_category7.append(np.array([566.25]))
    titles.append('(f) Reddit')

    # data_category_log1 = [np.log(arr) for arr in data_category1]
    # data_category_log2 = [np.log(arr) for arr in data_category2]
    # data_category_log3 = [np.log(arr) for arr in data_category3]
    # data_category_log4 = [np.log(arr) for arr in data_category4]
    # data_category_log5 = [np.log(arr) for arr in data_category5]
    # data_category_log6 = [np.log(arr) for arr in data_category6]
    # data_category_log7 = [np.log(arr) for arr in data_category7]
    # Set the width of the bars
    bar_width = 0.153
    coloumn_width = 1

    # Set the positions of the bars on the x-axis
    r1 = np.arange(len(data_category1[0])) * coloumn_width
    print(r1)
    r2 = [x + bar_width for x in r1]
    r3 = [x + bar_width for x in r2]
    r4 = [x + bar_width for x in r3]
    r5 = [x + bar_width for x in r4]
    r6 = [x + bar_width for x in r5]
    r7 = [2 * r1[-1] - r1[-2]]
    fig, axes = plt.subplots(nrows = 1, ncols = 3, figsize=(22,5))
    # plt.style.use('fivethirtyeight')
    i = 0

    # sns.set_palette("Set1")
    # sns.set_style("white")
    plt.rcParams['font.sans-serif'] ='Times New Roman'
    plt.rcParams["mathtext.fontset"]='stix'
    colors = ["#4E79A7", "#A0CBE8", "#F28E2B", "#FFBE7D", "#59A14F", "#8CD17D", "#B6992D", "#F1CE63", "#499894",
              "#86BCB6", "#E15759", "#E19D9A"]
    # Create the bar chart
    for ax in fig.axes:

        ax.bar(r1, data_category1[i], edgecolor = 'black', linewidth = 0.8,width=bar_width, label=labels[0], zorder=10)
        ax.bar(r2, data_category3[i], edgecolor = 'black', linewidth = 0.8,width=bar_width, label=labels[1], zorder=10)
        ax.bar(r3, data_category3[i], edgecolor = 'black', linewidth = 0.8,width=bar_width, label=labels[2], zorder=10)
        ax.bar(r4, data_category4[i], edgecolor = 'black', linewidth = 0.8,width=bar_width, label=labels[3], zorder=10)
        ax.bar(r5, data_category5[i], edgecolor = 'black', linewidth = 0.8,width=bar_width, label=labels[4], zorder=10)
        ax.bar(r6, data_category6[i], edgecolor = 'black', linewidth = 0.8,width=bar_width, label=labels[5], zorder=10)
        ax.bar(r7, data_category7[i], edgecolor = 'black', linewidth = 0.8,width=bar_width, label=labels[6], zorder=10)
    # Set the x-axis labels and title
        ax.set_xticks([0.36, 1.36, 2])
        ax.tick_params(bottom=False, top=False, left=False, right=False)
        ax.set_yticklabels(ax.get_yticks(), fontsize=16)
        ax.set_xticklabels(categories[i], fontsize=16)
        ax.set_xlabel('Reduction ratio', fontsize=18)
        ax.set_ylabel('Memory $(MB)$', fontsize=18)
        ax.set_title(titles[i], fontsize = 20, y = -0.30)
        ax.grid(True, alpha=0.5,  axis='y',color='gray', linewidth=1.5, linestyle='--')
        # ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        fontsize1 = 12
        for ii, data in enumerate(data_category1[i]):
            ax.text(ii, data, str(data), ha='center', va='bottom', zorder = 11, fontsize = fontsize1)
        for ii, data in enumerate(data_category2[i]):
            ax.text(ii+bar_width*1, data, str(data), ha='center', va='bottom', zorder = 11, fontsize = fontsize1)
        for ii, data in enumerate(data_category3[i]):
            ax.text(ii+bar_width*2, data, str(data), ha='center', va='bottom', zorder = 11, fontsize = fontsize1)
        for ii, data in enumerate(data_category4[i]):
            ax.text(ii+bar_width*3, data, str(data), ha='center', va='bottom', zorder = 11, fontsize = fontsize1)
        for ii, data in enumerate(data_category5[i]):
            ax.text(ii+bar_width*4, data, str(data), ha='center', va='bottom', zorder = 11, fontsize = fontsize1)
        for ii, data in enumerate(data_category6[i]):
            ax.text(ii+bar_width*5, data, str(data), ha='center', va='bottom', zorder = 11, fontsize = fontsize1)
        for ii, data in enumerate(data_category7[i]):
            ax.text(ii+2, data, str(data), ha='center', va='bottom', zorder = 11, fontsize = fontsize1)


        arrow_length = 0.6  # Proportion of arrow length to the total distance
        arrow_params1 = {'arrowstyle': ']-', 'linewidth': 2.5, 'color': 'black', 'alpha': 0.3}
        arrow_params2 = {'arrowstyle': '->', 'linewidth': 2.5,'color': 'black', 'alpha': 0.3}
        for j in range(len(data_category1[i])):
            # ax.annotate('', xytext=(r1[j], max(data_category7[i])), xy=(r1[j], data_category1[i][j] + 0.05 * (max(data_category7[i])- data_category1[i][j])), arrowprops=arrow_params, annotation_clip=False)
            # ax.annotate('', xytext=(r2[j], max(data_category7[i])), xy=(r2[j], data_category2[i][j] + 0.05 * (max(data_category7[i])- data_category2[i][j])), arrowprops=arrow_params, annotation_clip=False)
            # ax.annotate('', xytext=(r3[j], max(data_category7[i])), xy=(r3[j], data_category3[i][j] + 0.05 * (max(data_category7[i])- data_category3[i][j])), arrowprops = arrow_params, annotation_clip = False)
            # ax.annotate('', xytext=(r4[j], max(data_category7[i])), xy=(r4[j], data_category4[i][j] + 0.05 * (max(data_category7[i])- data_category4[i][j])), arrowprops = arrow_params, annotation_clip = False)
            # ax.annotate('', xytext=(r5[j], max(data_category7[i])), xy=(r5[j], data_category5[i][j] + 0.05 * (max(data_category7[i])- data_category5[i][j])), arrowprops = arrow_params, annotation_clip = False)
            ax.annotate("" ,
                        xytext=(r6[j], max(data_category7[i]) * 0.99),
                        xy=(r6[j], data_category6[i][j] + 0.64 * (max(data_category7[i]) - data_category6[i][j])),
                        arrowprops=arrow_params1, annotation_clip=False)
            ax.annotate("%.1fx\nsmaller" % max(data_category7[i]/data_category6[i][j]), xytext=(r6[j]-0.120, data_category6[i][j] + 0.5 * (max(data_category7[i]) - data_category6[i][j])), xy=(r6[j], data_category6[i][j] + 0.04 * (max(data_category7[i])- data_category6[i][j])), arrowprops = arrow_params2, annotation_clip = False, fontsize = 15)
        # if i == 0:
        #     ax.set_ylim(0, 35)
        i = i + 1

    lines, labels = fig.axes[-1].get_legend_handles_labels()
    # Add a legend
    plt.subplots_adjust(top = 0.860, bottom = 0.2000, left = 0.050, right = 0.990, wspace=0.230, hspace=0.200)

    fig.legend(lines, labels, loc = 'upper center', fontsize = 18, ncol = 7)
    # plt.tight_layout()
    # Show the plot
    plt.savefig('graph_memory.pdf', bbox_inches='tight')
    # plt.show()
def f4():
    # import matplotlib
    # import matplotlib.pyplot as plt
    # import numpy as np
    # import matplotlib.ticker as ticker
    # # import seaborn as sns
    # palette = matplotlib.pyplot.get_cmap('Set3')
    # Example data
    categories = []
    categories.append(['0.16%', '0.32%', '100%'])
    categories.append(['0.10%', '0.50%', '100%'])
    categories.append(['0.10%', '0.50%', '100%'])
    labels = ['Random', 'Degree', 'Herding', 'K-Center', 'VNG', 'MCond', 'Whole']
    data_category1 = []
    data_category2 = []
    data_category3 = []
    data_category4 = []
    data_category5 = []
    data_category6 = []
    data_category7 = []
    titles = []
    #Pubmed
    data_category1.append(np.array([4.59, 5.31 ]))
    data_category2.append(np.array([4.72 , 5.42 ]))
    data_category3.append(np.array([4.23 , 4.64 ]))
    data_category4.append(np.array([4.30 , 4.64 ]))
    data_category5.append(np.array([8.37 , 10.52 ]))
    data_category6.append(np.array([7.39 , 8.78 ]))
    data_category7.append(np.array([143.20 ]))
    titles.append('(a) Pubmed')

    data_category1.append(np.array([5.73 , 7.46 ]))
    data_category2.append(np.array([6.05 , 8.04 ]))
    data_category3.append(np.array([5.83 , 7.50 ]))
    data_category4.append(np.array([5.59 , 7.44 ]))
    data_category5.append(np.array([11.41 , 25.73 ]))
    data_category6.append(np.array([13.89 , 31.38 ]))
    data_category7.append(np.array([656.42 ]))
    titles.append('(b) Flickr')

    data_category1.append(np.array([54.63 , 57.94 ]))
    data_category2.append(np.array([56.61 , 62.38 ]))
    data_category3.append(np.array([54.58 , 57.90 ]))
    data_category4.append(np.array([54.49 , 57.93 ]))
    data_category5.append(np.array([199.69 , 1265.13 ]))
    data_category6.append(np.array([150.09 , 449.12 ]))
    data_category7.append(np.array([18242.64 ]))
    titles.append('(c) Reddit')

    # Set the width of the bars
    # data_category_log1 = [np.log(arr) for arr in data_category1]
    # data_category_log2 = [np.log(arr) for arr in data_category2]
    # data_category_log3 = [np.log(arr) for arr in data_category3]
    # data_category_log4 = [np.log(arr) for arr in data_category4]
    # data_category_log5 = [np.log(arr) for arr in data_category5]
    # data_category_log6 = [np.log(arr) for arr in data_category6]
    # data_category_log7 = [np.log(arr) for arr in data_category7]
    # Set the width of the bars
    bar_width = 0.153
    coloumn_width = 1

    # Set the positions of the bars on the x-axis
    r1 = np.arange(len(data_category1[0])) * coloumn_width
    print(r1)
    r2 = [x + bar_width for x in r1]
    r3 = [x + bar_width for x in r2]
    r4 = [x + bar_width for x in r3]
    r5 = [x + bar_width for x in r4]
    r6 = [x + bar_width for x in r5]
    r7 = [2 * r1[-1] - r1[-2]]
    fig, axes = plt.subplots(nrows = 1, ncols = 3, figsize=(22,5), dpi=300)
    # plt.style.use('fivethirtyeight')
    i = 0

    # sns.set_palette("Set1")
    # sns.set_style("white")
    plt.rcParams['font.sans-serif'] ='Times New Roman'
    plt.rcParams["mathtext.fontset"]='stix'
    colors = ["#4E79A7", "#A0CBE8", "#F28E2B", "#FFBE7D", "#59A14F", "#8CD17D", "#B6992D", "#F1CE63", "#499894",
              "#86BCB6", "#E15759", "#E19D9A"]
    # Create the bar chart
    for ax in fig.axes:

        ax.bar(r1, data_category1[i],  edgecolor = 'black', linewidth = 0.8,width=bar_width, label=labels[0], zorder=10)
        ax.bar(r2, data_category3[i],  edgecolor = 'black', linewidth = 0.8,width=bar_width, label=labels[1], zorder=10)
        ax.bar(r3, data_category3[i],  edgecolor = 'black', linewidth = 0.8,width=bar_width, label=labels[2], zorder=10)
        ax.bar(r4, data_category4[i],  edgecolor = 'black', linewidth = 0.8,width=bar_width, label=labels[3], zorder=10)
        ax.bar(r5, data_category5[i],  edgecolor = 'black', linewidth = 0.8,width=bar_width, label=labels[4], zorder=10)
        ax.bar(r6, data_category6[i],  edgecolor = 'black', linewidth = 0.8,width=bar_width, label=labels[5], zorder=10)
        ax.bar(r7, data_category7[i],  edgecolor = 'black', linewidth = 0.8,width=bar_width, label=labels[6], zorder=10)
    # Set the x-axis labels and title
        ax.set_xticks([0.36, 1.36, 2])
        ax.tick_params(bottom=False, top=False, left=False, right=False)
        ax.set_yticklabels(ax.get_yticks(), fontsize=16)
        ax.set_xticklabels(categories[i], fontsize=16)
        ax.set_xlabel('Reduction ratio', fontsize=18)
        ax.set_ylabel('Time $(ms)$', fontsize=18)
        ax.set_title(titles[i], fontsize = 20, y = -0.30)
        ax.grid(True, linestyle='dotted', linewidth=1)
        # ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        fontsize1 = 9
        for ii, data in enumerate(data_category1[i]):
            ax.text(ii, data, str(data), ha='center', va='bottom', zorder = 11, fontsize = fontsize1)
        for ii, data in enumerate(data_category2[i]):
            ax.text(ii+bar_width*1, data, str(data), ha='center', va='bottom', zorder = 11, fontsize = fontsize1)
        for ii, data in enumerate(data_category3[i]):
            ax.text(ii+bar_width*2, data, str(data), ha='center', va='bottom', zorder = 11, fontsize = fontsize1)
        for ii, data in enumerate(data_category4[i]):
            ax.text(ii+bar_width*3, data, str(data), ha='center', va='bottom', zorder = 11, fontsize = fontsize1)
        for ii, data in enumerate(data_category5[i]):
            ax.text(ii+bar_width*4, data, str(data), ha='center', va='bottom', zorder = 11, fontsize = fontsize1)
        for ii, data in enumerate(data_category6[i]):
            ax.text(ii+bar_width*5, data, str(data), ha='center', va='bottom', zorder = 11, fontsize =fontsize1)
        for ii, data in enumerate(data_category7[i]):
            ax.text(ii+2, data, str(data), ha='center', va='bottom', zorder = 11, fontsize = fontsize1)


        arrow_length = 0.6  # Proportion of arrow length to the total distance
        arrow_params1 = {'arrowstyle': ']-', 'linewidth': 2.5, 'color': 'black', 'alpha': 0.3}
        arrow_params2 = {'arrowstyle': '->', 'linewidth': 2.5,'color': 'black', 'alpha': 0.3}
        for j in range(len(data_category1[i])):
            # ax.annotate('', xytext=(r1[j], max(data_category7[i])), xy=(r1[j], data_category1[i][j] + 0.05 * (max(data_category7[i])- data_category1[i][j])), arrowprops=arrow_params, annotation_clip=False)
            # ax.annotate('', xytext=(r2[j], max(data_category7[i])), xy=(r2[j], data_category2[i][j] + 0.05 * (max(data_category7[i])- data_category2[i][j])), arrowprops=arrow_params, annotation_clip=False)
            # ax.annotate('', xytext=(r3[j], max(data_category7[i])), xy=(r3[j], data_category3[i][j] + 0.05 * (max(data_category7[i])- data_category3[i][j])), arrowprops = arrow_params, annotation_clip = False)
            # ax.annotate('', xytext=(r4[j], max(data_category7[i])), xy=(r4[j], data_category4[i][j] + 0.05 * (max(data_category7[i])- data_category4[i][j])), arrowprops = arrow_params, annotation_clip = False)
            # ax.annotate('', xytext=(r5[j], max(data_category7[i])), xy=(r5[j], data_category5[i][j] + 0.05 * (max(data_category7[i])- data_category5[i][j])), arrowprops = arrow_params, annotation_clip = False)
            ax.annotate("" ,
                        xytext=(r6[j], max(data_category7[i]) * 0.99),
                        xy=(r6[j], data_category6[i][j] + 0.64 * (max(data_category7[i]) - data_category6[i][j])),
                        arrowprops=arrow_params1, annotation_clip=False)
            if max(data_category7[i]/data_category6[i][j]) > 100:
                ax.annotate("%.1fx\nfaster" % max(data_category7[i] / data_category6[i][j]), xytext=(
                r6[j] - 0.112, data_category6[i][j] + 0.5 * (max(data_category7[i]) - data_category6[i][j])),
                            xy=(r6[j], data_category6[i][j] + 0.04 * (max(data_category7[i]) - data_category6[i][j])),
                            arrowprops=arrow_params2, annotation_clip=False, fontsize=15)
            else:
                ax.annotate("%.1fx\nfaster" % max(data_category7[i]/data_category6[i][j]), xytext=(r6[j]-0.092, data_category6[i][j] + 0.5 * (max(data_category7[i]) - data_category6[i][j])), xy=(r6[j], data_category6[i][j] + 0.04 * (max(data_category7[i])- data_category6[i][j])), arrowprops = arrow_params2, annotation_clip = False, fontsize = 15)
        # if i == 0:
        #     ax.set_ylim(0, 35)
        i = i + 1

    lines, labels = fig.axes[-1].get_legend_handles_labels()
    # Add a legend
    plt.subplots_adjust(top = 0.860, bottom = 0.2000, left = 0.050, right = 0.990, wspace=0.230, hspace=0.200)

    fig.legend(lines, labels, loc = 'upper center', fontsize = 18, ncol = 7)
    # plt.tight_layout()
    # Show the plot
    plt.savefig('graph_time.pdf', bbox_inches='tight')
    # plt.show()

def f5():
    import torch
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns
    initial=torch.load('loss_initial.pt')
    noinitial=torch.load('loss_noinitial.pt')

    x1 = []
    y1 = []
    x2 = []
    y2 = []

    for i in range(501,2001):
        x1.append(initial[i][0])
        y1.append(initial[i][1])
        x2.append(noinitial[i][0])
        y2.append(noinitial[i][1])
    x1 = np.array(x1)
    x2 = np.array(x2)
    y1 = np.array(y1)
    y2 = np.array(y2)
    x1 -= 500
    x2 -= 500


    fontsize=20

    sns.set_palette("Set1")
    sns.set_style("white")
    plt.figure(figsize=(8,4), dpi=300)

    plt.tick_params(labelsize=fontsize)
    plt.rcParams['font.sans-serif'] ='Times New Roman'
    plt.rcParams["mathtext.fontset"]='stix'
    plt.rcParams['font.size'] = fontsize

    sns.lineplot(x=x1,y=y1,linewidth=1,label="w/ initializaion")
    sns.lineplot(x=x2,y=y2,linewidth=1,label="w/o initializaion")

    plt.grid(True, linestyle='dotted', linewidth=1)

    plt.legend(loc=1, fontsize=fontsize)
    plt.ylabel('Loss', fontsize=fontsize)
    plt.xlabel('Epoch', fontsize=fontsize)

    plt.savefig('loss.png',bbox_inches='tight')
    plt.savefig('loss.pdf',bbox_inches='tight')

def f6():
    import seaborn as sns
    import matplotlib.ticker as ticker
    import matplotlib.pyplot as plt
    import numpy as np
    # plt.rcParams['font.family'] = 'Times New Roman'
    # Example data
    bar_data = []
    line_data = []
    x = []
    x_labels = []
    barmax = []
    barmin = []
    linemax = []
    linemin = []
    titles = []
    titles.append('(a) Pubmed (0.32%)')
    titles.append('(b) Flickr (0.50%)')
    titles.append('(c) Reddit (0.10%)')
    x_labels.append([0,
    0.001,
    0.003,
    0.005,
    0.007,
    0.01,
    0.03,
    0.05,
    0.07,
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7])
    x.append(np.arange(1, len(x_labels[0])+1))
    barmax.append(0.80)
    barmin.append(0.75)
    linemax.append(1.00)
    linemin.append(0.00)
    bar_data.append([ 77.07, 77.07, 77.50, 77.47, 77.53, 77.63, 77.63, 77.93, 77.87, 76.27, 76.27, 76.23, 76.13, 76.13, 76.23, 76.33])
    line_data.append([
                 1,
                 0.99416,
                 0.70245,
                 0.48062,
                 0.39181,
                 0.32489,
                 0.18218,
                 0.09384,
                 0.04277,
                 0.01783,
                 0.00293,
                 0.00103,
                 0.00053,
                 0.00029,
                 0.00018,
                 0.00012])

    x_labels.append([0,
    0.000001,
    0.00001,
    0.0001,
    0.001,
    0.01,
    0.03,
    0.05,
    0.06,
    0.07,
    0.08,
    0.09,
    0.1,
    0.11,
    0.12,
    0.2,
    0.3,
    0.4])
    x.append(np.arange(1, len(x_labels[1])+1))
    barmax.append(0.50)
    barmin.append(0.40)
    linemax.append(0.50)
    linemin.append(0.00)
    bar_data.append([
    46.92,
    46.92,
    47.03,
    47.03,
    47.16,
    47.31,
    47.57,
    47.93,
    48.22,
    48.07,
    47.81,
    47.43,
    47.07,
    46.79,
    46.34,
    44.66,
    43.33,
    43.14])
    line_data.append([
    0.47571,
    0.39625,
    0.29933,
    0.17768,
    0.12603,
    0.08644,
    0.05574,
    0.03492,
    0.02552,
    0.01796,
    0.01247,
    0.0088,
    0.0063,
    0.00463,
    0.00348,
    0.00067,
    0.00019,
    0.00008])

    x_labels.append([
    0,
    0.00001,
    0.0001,
    0.001,
    0.01,
    0.1,
    0.15,
    0.17,
    0.18,
    0.2,
    0.22,
    0.24,
    0.25,
    0.3,
    0.4,
    0.5])
    print(len(x_labels[2]))
    x.append(np.arange(1, len(x_labels[2])+1))
    print(len(x[2]))
    barmax.append(0.905)
    barmin.append(0.87)
    linemax.append(0.35)
    linemin.append(0.00)
    bar_data.append([
    90.17,
    90.17,
    90.17,
    90.17,
    90.27,
    90.27,
    90.28,
    90.29,
    90.3,
    90.32,
    90.34,
    90.34,
    90.33,
    90.29,
    90.05,
    87.77])
    line_data.append([
    0.31,
    0.194,
    0.174,
    0.168,
    0.025,
    0.015,
    0.013,
    0.012,
    0.012,
    0.011,
    0.01,
    0.01,
    0.009,
    0.008,
    0.005,
    0.002])

    bar_data = [[x / 100 for x in arr] for arr in bar_data]
    # Create a custom color palette for the bar chart and line graph
    # bar_color = sns.color_palette("cmap")[3]
    # line_color = sns.color_palette("Reds")[3]

    # Create a figure and axes
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize = (50, 5))
    aspect_ratio = 0.1  # Adjust this value to change the aspect ratio

    # Set the figure size based on the aspect ratio
    sns.set(rc={'figure.figsize': (5 * aspect_ratio, 5)})
    plt.rcParams['font.sans-serif'] ='Times New Roman'
    plt.rcParams["mathtext.fontset"]='stix'
    sns.set_context("notebook", rc={"xtick.labelsize": 24, "ytick.labelsize": 24})
    sns.set_palette("Set1")
    current_palette = sns.color_palette("Set1")
    sns.set_style("white")
    # Draw the bar chart with custom color
    i = 0
    # colors = ["#4E79A7", "#A0CBE8", "#F28E2B", "#FFBE7D", "#59A14F", "#8CD17D", "#B6992D", "#F1CE63", "#499894",
    #           "#86BCB6", "#E15759", "#E19D9A"]
    for i in range(len(x)):
        print(x[i], bar_data[i])
        xx = np.array(x[i], dtype= int)
        yy1 = np.array(bar_data[i], dtype= float)
        yy2 = np.array(line_data[i], dtype= float)
        print(xx, yy1)
        sub1 = sns.barplot(x=xx, y=yy1, edgecolor = 'black', linewidth = 0.6,alpha = 0.6, color= current_palette[1], ax = axes[i], label = 'Accuracy')
        ax2 = axes[i].twinx()
        sub2  = sns.lineplot(x=xx-1, y=yy2, marker='^',markersize = 10, linewidth = 2.5, alpha = 1, color=current_palette[4], ax = ax2, label = 'Sparsity')
        axes[i].set_xticklabels(x_labels[i], fontsize= 15, rotation=45)
        # axes[i].set_yticklabels([0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=14)
        # axes[i].tick_params(axis='y', labelsize=16)
        axes[i].set_xlabel('$\delta$', fontsize=19, fontname = 'Times New Roman')
        axes[i].set_ylabel('Accuracy', fontsize=16, fontname = 'Times New Roman')
        len1 = barmax[i] - barmin[i]
        axes[i].set_ylim(barmin[i]- 0.02 * len1, barmax[i]+ 0.02 * len1 )
        ax2.set_ylabel('Sparsity', fontsize = 16, fontname = 'Times New Roman')
        len2 = linemax[i] - linemin[i]
        ax2.set_ylim(linemin[i]- 0.02 * len2, linemax[i] + 0.02 * len2)
        axes[i].tick_params(axis='x', labelsize=14)
        axes[i].tick_params(axis='y', labelsize=14)
        ax2.tick_params(axis='y', labelsize=14)
        axes[i].tick_params(bottom=False, top=False, left=True, right=True)
        ax2.tick_params(bottom=False, top=False, left=True, right=True)
        axes[i].set_title(titles[i], fontsize = 18, y = -0.45, fontname = 'Times New Roman')
        axes[i].grid(True, linestyle='dotted', linewidth=1)
        if i == 2:
            axes[i].yaxis.set_major_locator(ticker.MaxNLocator(nbins=8))
            ax2.yaxis.set_major_locator(ticker.MaxNLocator(nbins=8))
        axes[i].set_clip_on(False)
        ax2.set_clip_on(False)
        # sns.legend.remove()
        axes[i].legend().remove()
        ax2.legend().remove()


    lines1, labels1 = sub1.get_legend_handles_labels()
    lines2, labels2 = sub2.get_legend_handles_labels()
    lines = lines1 + lines2
    labels = labels1 + labels2
    # Add a legend
    plt.subplots_adjust(top = 0.860, bottom = 0.260, left = 0.040, right = 0.955, wspace=0.3450, hspace=0.200)
    legend_font = {'family': 'Times New Roman', 'size': 18}
    fig.legend(lines, labels, loc = 'upper center', ncol = 2, prop = legend_font )
    # Show the plot
    plt.show()
    plt.savefig('sparse.pdf', bbox_inches='tight')

def f7():
    # import seaborn as sns
    # import matplotlib.pyplot as plt
    # import numpy as np
    # plt.rcParams['font.sans-serif'] ='Times New Roman'
    # plt.rcParams["mathtext.fontset"]='stix'
    # Example data
    bar_data = []
    line_data = []
    x = []
    x_labels = []
    barmax = []
    barmin = []
    titles = []
    titles.append('(a)')
    titles.append('(b)')
    labels= []
    labels.append(' $\lambda}$ ')
    labels.append(' $\\beta$ ')
    x_labels.append([0,
    0.01,
    0.1,
    1,
    10,
    100,
    1000])
    x.append(np.arange(1, len(x_labels[0])+1))
    barmax.append(0.48)
    barmin.append(0.46)
    bar_data.append([47.04,
    47.65,
    47.46,
    47.31,
    47.52,
    47.13,
    47.01])

    x_labels.append([0,
    0.01,
    0.1,
    1,
    10,
    100,
    1000])
    x.append(np.arange(1, len(x_labels[1])+1))
    barmax.append(0.48)
    barmin.append(0.46)
    bar_data.append([46.82,
    46.87,
    46.79,
    46.66,
    47.53,
    47.88,
    47.59])


    bar_data = [[x / 100 for x in arr] for arr in bar_data]
    # Create a custom color palette for the bar chart and line graph
    # bar_color = sns.color_palette("cmap")[3]
    # line_color = sns.color_palette("Reds")[3]

    # Create a figure and axes
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize = (10, 4.5))
    aspect_ratio = 0.1  # Adjust this value to change the aspect ratio
    sns.set(font = 'Times New Roman')
    # Set the figure size based on the aspect ratio
    sns.set(rc={'figure.figsize': (5 * aspect_ratio, 5)})
    # plt.rcParams['font.sans-serif'] ='Times New Roman'
    # plt.rcParams["mathtext.fontset"]='stix'
    sns.set_context( rc={"xtick.labelsize": 24, "ytick.labelsize": 24})
    sns.set_palette("Set1")
    current_palette = sns.color_palette("Set1")
    sns.set_style("white")
    # Draw the bar chart with custom color
    i = 0
    # colors = ["#4E79A7", "#A0CBE8", "#F28E2B", "#FFBE7D", "#59A14F", "#8CD17D", "#B6992D", "#F1CE63", "#499894",
    #           "#86BCB6", "#E15759", "#E19D9A"]
    for i in range(len(x)):
        xx = np.array(x[i], dtype= int)
        yy1 = np.array(bar_data[i], dtype= float)
        sub1 = sns.barplot(x=xx, y=yy1, edgecolor = 'black', linewidth = 1,alpha = 0.8, color= current_palette[1], ax = axes[i], label = 'Accuracy', hatch = '////')
        axes[i].set_xticklabels(x_labels[i], fontsize= 18, rotation=45, fontname = 'Times New Roman')
        axes[i].set_yticklabels([0,'0.460', '0.465', '0.470', '0.475', '0.480'], fontsize=18, fontname='Times New Roman')
        axes[i].set_xlabel(labels[i], fontsize=24, fontname = 'Times New Roman')
        axes[i].set_ylabel('Accuracy', fontsize=20, fontname = 'Times New Roman')
        len1 = barmax[i] - barmin[i]
        axes[i].set_ylim(barmin[i]- 0.01 * len1, barmax[i]+ 0.01 * len1 )
        axes[i].tick_params(axis='x', labelsize=18)
        axes[i].tick_params(axis='y', labelsize=18)
        axes[i].tick_params(bottom=False, top=False, left=False, right=False)
        # axes[i].set_title(titles[i], fontsize = 22, y = -0.50)
        axes[i].grid(True, linestyle='dotted', linewidth=1)
        # sns.legend.remove()
        axes[i].legend().remove()


    lines1, labels1 = sub1.get_legend_handles_labels()
    # Add a legend
    plt.subplots_adjust(top = 0.950, bottom = 0.240, left = 0.110, right = 0.99, wspace=0.3400, hspace=0.200)

    # fig.legend(lines1, labels1, loc = 'upper center', fontsize = 18, ncol = 2)
    # Show the plot
    plt.show()
    plt.savefig('ablation.pdf', bbox_inches='tight')


def f8():

    # palette = matplotlib.pyplot.get_cmap('Set3')
    # Example data
    categories = []
    categories.append(['0.16%', '0.32%', '100%'])
    categories.append(['0.10%', '0.50%', '100%'])
    categories.append(['0.10%', '0.50%', '100%'])
    labels = ['Random', 'Degree', 'Herding', 'K-Center', 'VNG', 'MCond', 'Whole']
    data_category1 = []
    data_category2 = []
    data_category3 = []
    data_category4 = []
    data_category5 = []
    data_category6 = []
    data_category7 = []
    titles = []
    #Pubmed
    data_category1.append(np.array([4.17, 4.83]))
    data_category2.append(np.array([4.45, 4.95]))
    data_category3.append(np.array([3.67, 4.02]))
    data_category4.append(np.array([3.63, 4.17]))
    data_category5.append(np.array([7.17, 9.20]))
    data_category6.append(np.array([5.80, 6.96]))
    data_category7.append(np.array([135.85]))
    titles.append("(a) Pubmed")

    data_category1.append(np.array([5.39, 7.38]))
    data_category2.append(np.array([5.44, 7.82]))
    data_category3.append(np.array([5.12, 7.35]))
    data_category4.append(np.array([5.16, 7.34]))
    data_category5.append(np.array([11.18, 24.98]))
    data_category6.append(np.array([12.22, 30.83]))
    data_category7.append(np.array([631.03]))
    titles.append("(b) Flickr")

    data_category1.append(np.array([51.72, 53.62]))
    data_category2.append(np.array([51.04, 59.37]))
    data_category3.append(np.array([48.90, 50.88]))
    data_category4.append(np.array([48.70, 51.55]))
    data_category5.append(np.array([194.25, 1160.80]))
    data_category6.append(np.array([144.73, 416.67]))
    data_category7.append(np.array([17303.18 ]))
    titles.append("(c) Reddit")

    # Set the width of the bars
    # data_category_log1 = [np.log(arr) for arr in data_category1]
    # data_category_log2 = [np.log(arr) for arr in data_category2]
    # data_category_log3 = [np.log(arr) for arr in data_category3]
    # data_category_log4 = [np.log(arr) for arr in data_category4]
    # data_category_log5 = [np.log(arr) for arr in data_category5]
    # data_category_log6 = [np.log(arr) for arr in data_category6]
    # data_category_log7 = [np.log(arr) for arr in data_category7]
    # Set the width of the bars
    bar_width = 0.12
    coloumn_width = 1

    # Set the positions of the bars on the x-axis
    r1 = np.arange(len(data_category1[0])) * coloumn_width
    print(r1)
    r2 = [x + bar_width for x in r1]
    r3 = [x + bar_width for x in r2]
    r4 = [x + bar_width for x in r3]
    r5 = [x + bar_width for x in r4]
    r6 = [x + bar_width for x in r5]
    r7 = [2 * r1[-1] - r1[-2]]
    fig, axes = plt.subplots(nrows = 1, ncols = 6, figsize=(30,7))
    # plt.style.use('fivethirtyeight')
    i = 0
    # sns.set_palette("Set1")
    # sns.set_style("white")
    plt.rcParams['font.sans-serif'] ='Times New Roman'
    plt.rcParams["mathtext.fontset"]='stix'
    # colors = ["#4E79A7", "#A0CBE8", "#F28E2B", "#FFBE7D", "#59A14F", "#8CD17D", "#B6992D", "#F1CE63", "#499894",
    #           "#86BCB6", "#E15759", "#E19D9A"]
    # Create the bar chart
    for i in range(3):

        axes[i].barh(r1, data_category1[i], height = bar_width, edgecolor = 'black', linewidth = 0.8, label=labels[0], zorder=10)
        axes[i].barh(r2, data_category3[i], height = bar_width,edgecolor = 'black', linewidth = 0.8, label=labels[1], zorder=10)
        axes[i].barh(r3, data_category3[i], height = bar_width,edgecolor = 'black', linewidth = 0.8, label=labels[2], zorder=10)
        axes[i].barh(r4, data_category4[i], height = bar_width,edgecolor = 'black', linewidth = 0.8, label=labels[3], zorder=10)
        axes[i].barh(r5, data_category5[i], height = bar_width,edgecolor = 'black', linewidth = 0.8, label=labels[4], zorder=10)
        axes[i].barh(r6, data_category6[i], height = bar_width,edgecolor = 'black', linewidth = 0.8, label=labels[5], zorder=10)
        axes[i].barh(r7, data_category7[i], height = bar_width,edgecolor = 'black', linewidth = 0.8, label=labels[6], zorder=10)
    # Set the x-axis labels and title
        axes[i].set_yticks([0.36, 1.36, 2])
        axes[i].tick_params(bottom=False, top=False, left=False, right=False)
        if i == 2:
            axes[i].xaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
            # axes[i].set_xticklabels([0.0, 5000.0, 10000.0, 15000.0, 20000.0], fontsize=16)
            axes[i].set_xticklabels(axes[i].get_xticks(), fontsize=16)
        else:
            axes[i].set_xticklabels(axes[i].get_xticks(), fontsize=16)
        axes[i].set_yticklabels(categories[i], fontsize=16)
        axes[i].set_ylabel('Reduction ratio', fontsize=18)
        axes[i].set_xlabel('Time $(ms)$', fontsize=18)
        axes[i].set_title(titles[i], fontsize = 20, y = -0.20)
        axes[i].grid(True, alpha=0.5,  axis='x',color='gray', linewidth=1.5, linestyle='--')
        # ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        # fontsize1 = 10
        # for ii, data in enumerate(data_category1[i]):
        #     axes[i].text(ii, data, str(data), ha='center', va='bottom', zorder = 11, fontsize = fontsize1)
        # for ii, data in enumerate(data_category2[i]):
        #     axes[i].text(ii+bar_width*1, data, str(data), ha='center', va='bottom', zorder = 11, fontsize = fontsize1)
        # for ii, data in enumerate(data_category3[i]):
        #     axes[i].text(ii+bar_width*2, data, str(data), ha='center', va='bottom', zorder = 11, fontsize = fontsize1)
        # for ii, data in enumerate(data_category4[i]):
        #     axes[i].text(ii+bar_width*3, data, str(data), ha='center', va='bottom', zorder = 11, fontsize = fontsize1)
        # for ii, data in enumerate(data_category5[i]):
        #     axes[i].text(ii+bar_width*4, data, str(data), ha='center', va='bottom', zorder = 11, fontsize = fontsize1)
        # for ii, data in enumerate(data_category6[i]):
        #     axes[i].text(ii+bar_width*5, data, str(data), ha='center', va='bottom', zorder = 11, fontsize =fontsize1)
        # for ii, data in enumerate(data_category7[i]):
        #     axes[i].text(ii+2, data, str(data), ha='center', va='bottom', zorder = 11, fontsize = fontsize1)



        arrow_length = 0.6  # Proportion of arrow length to the total distance
        arrow_params1 = {'arrowstyle': ']-', 'linewidth': 2.5, 'color': 'black', 'alpha': 0.3}
        arrow_params2 = {'arrowstyle': '->', 'linewidth': 2.5,'color': 'black', 'alpha': 0.3}
        for j in range(len(data_category1[i])):
            # ax.annotate('', xytext=(r1[j], max(data_category7[i])), xy=(r1[j], data_category1[i][j] + 0.05 * (max(data_category7[i])- data_category1[i][j])), arrowprops=arrow_params, annotation_clip=False)
            # ax.annotate('', xytext=(r2[j], max(data_category7[i])), xy=(r2[j], data_category2[i][j] + 0.05 * (max(data_category7[i])- data_category2[i][j])), arrowprops=arrow_params, annotation_clip=False)
            # ax.annotate('', xytext=(r3[j], max(data_category7[i])), xy=(r3[j], data_category3[i][j] + 0.05 * (max(data_category7[i])- data_category3[i][j])), arrowprops = arrow_params, annotation_clip = False)
            # ax.annotate('', xytext=(r4[j], max(data_category7[i])), xy=(r4[j], data_category4[i][j] + 0.05 * (max(data_category7[i])- data_category4[i][j])), arrowprops = arrow_params, annotation_clip = False)
            # ax.annotate('', xytext=(r5[j], max(data_category7[i])), xy=(r5[j], data_category5[i][j] + 0.05 * (max(data_category7[i])- data_category5[i][j])), arrowprops = arrow_params, annotation_clip = False)
            if max(data_category7[i] / data_category6[i][j]) > 100:
                axes[i].annotate("" ,
                            xytext=( max(data_category7[i]) * 0.99, r6[j]),
                            xy=( data_category6[i][j] + 0.66 * (max(data_category7[i]) - data_category6[i][j]), r6[j]),
                            arrowprops=arrow_params1, annotation_clip=False)
            else:
                axes[i].annotate("" ,
                            xytext=( max(data_category7[i]) * 0.99, r6[j]),
                            xy=( data_category6[i][j] + 0.64 * (max(data_category7[i]) - data_category6[i][j]), r6[j]),
                            arrowprops=arrow_params1, annotation_clip=False)
            axes[i].annotate("%.1fx" % max(data_category7[i] / data_category6[i][j]), xytext=(data_category6[i][j] + 0.5 * (max(data_category7[i]) - data_category6[i][j])
            ,r6[j] - 0.03 ),
                        xy=(data_category6[i][j] + 0.04 * (max(data_category7[i]) - data_category6[i][j]), r6[j]),
                        arrowprops=arrow_params2, annotation_clip=False, fontsize=15)
        # if i == 0:
        #     ax.set_ylim(0, 35)

    # lines, labels = fig.axes[-1].get_legend_handles_labels()
    # Add a legend
    # plt.subplots_adjust(top = 0.860, bottom = 0.2000, left = 0.050, right = 0.990, wspace=0.230, hspace=0.200)
    #
    # fig.legend(lines, labels, loc = 'upper center', fontsize = 18, ncol = 7)
    # plt.tight_layout()
    # Show the plot
    # plt.savefig('node_time.pdf', bbox_inches='tight')
    # plt.show()
    # import matplotlib
    # import matplotlib.pyplot as plt
    # import numpy as np
    # import matplotlib.ticker as ticker
    # import seaborn as sns
    # palette = matplotlib.pyplot.get_cmap('Set3')
    # Example data
    categories = []
    categories.append(['0.16%', '0.32%', '100%'])
    categories.append(['0.10%', '0.50%', '100%'])
    categories.append(['0.10%', '0.50%', '100%'])
    labels = ['Random', 'Degree', 'Herding', 'K-Center', 'VNG', 'MCond', 'Whole']
    data_category1 = []
    data_category2 = []
    data_category3 = []
    data_category4 = []
    data_category5 = []
    data_category6 = []
    data_category7 = []
    titles = []
    #Pubmed
    data_category1.append(np.array([1.98, 2.04]))
    data_category2.append(np.array([1.99, 2.05]))
    data_category3.append(np.array([1.98, 2.05]))
    data_category4.append(np.array([1.98, 2.04]))
    data_category5.append(np.array([2.48, 2.61]))
    data_category6.append(np.array([2.55, 3.88]))
    data_category7.append(np.array([38.62]))
    titles.append('(d) Pubmed')

    data_category1.append(np.array([2.01, 2.36]))
    data_category2.append(np.array([2.02, 2.39]))
    data_category3.append(np.array([2.01, 2.36]))
    data_category4.append(np.array([2.01, 2.36]))
    data_category5.append(np.array([3.04, 4.26]))
    data_category6.append(np.array([5.01, 9.20]))
    data_category7.append(np.array([133.30]))
    titles.append('(e) Flickr')

    data_category1.append(np.array([2.67, 4.11]))
    data_category2.append(np.array([2.76, 5.12]))
    data_category3.append(np.array([2.67, 4.11]))
    data_category4.append(np.array([2.67, 4.10]))
    data_category5.append(np.array([7.33, 19.98]))
    data_category6.append(np.array([10.12, 12.12]))
    data_category7.append(np.array([565.94]))
    titles.append('(f) Reddit')

    # data_category_log1 = [np.log(arr) for arr in data_category1]
    # data_category_log2 = [np.log(arr) for arr in data_category2]
    # data_category_log3 = [np.log(arr) for arr in data_category3]
    # data_category_log4 = [np.log(arr) for arr in data_category4]
    # data_category_log5 = [np.log(arr) for arr in data_category5]
    # data_category_log6 = [np.log(arr) for arr in data_category6]
    # data_category_log7 = [np.log(arr) for arr in data_category7]
    # Set the width of the bars
    bar_width = 0.12
    coloumn_width = 1

    # Set the positions of the bars on the x-axis
    r1 = np.arange(len(data_category1[0])) * coloumn_width
    print(r1)
    r2 = [x + bar_width for x in r1]
    r3 = [x + bar_width for x in r2]
    r4 = [x + bar_width for x in r3]
    r5 = [x + bar_width for x in r4]
    r6 = [x + bar_width for x in r5]
    r7 = [2 * r1[-1] - r1[-2]]
    # fig, axes = plt.subplots(nrows = 1, ncols = 3, figsize=(22,5))
    # plt.style.use('fivethirtyeight')
    i = 0

    plt.rcParams['font.sans-serif'] ='Times New Roman'
    plt.rcParams["mathtext.fontset"]='stix'
    # colors = ["#4E79A7", "#A0CBE8", "#F28E2B", "#FFBE7D", "#59A14F", "#8CD17D", "#B6992D", "#F1CE63", "#499894",
    #           "#86BCB6", "#E15759", "#E19D9A"]
    # Create the bar chart
    for i in range(3):
        axes[i+3].barh(r1, data_category1[i], edgecolor = 'black', linewidth = 0.8, height=bar_width, label=labels[0], zorder=10)
        axes[i+3].barh(r2, data_category3[i], edgecolor = 'black', linewidth = 0.8,height=bar_width, label=labels[1], zorder=10)
        axes[i+3].barh(r3, data_category3[i], edgecolor = 'black', linewidth = 0.8,height=bar_width, label=labels[2], zorder=10)
        axes[i+3].barh(r4, data_category4[i], edgecolor = 'black', linewidth = 0.8,height=bar_width, label=labels[3], zorder=10)
        axes[i+3].barh(r5, data_category5[i], edgecolor = 'black', linewidth = 0.8,height=bar_width, label=labels[4], zorder=10)
        axes[i+3].barh(r6, data_category6[i], edgecolor = 'black', linewidth = 0.8,height=bar_width, label=labels[5], zorder=10)
        axes[i+3].barh(r7, data_category7[i], edgecolor = 'black', linewidth = 0.8,height=bar_width, label=labels[6], zorder=10)
    # Set the x-axis labels and title
        axes[i+3].set_yticks([0.36, 1.36, 2])
        axes[i+3].tick_params(bottom=False, top=False, left=False, right=False)
        if i == 0:
            axes[i+3].xaxis.set_major_locator(ticker.MaxNLocator(nbins=6))
            # axes[i].set_xticklabels([0.0, 5000.0, 10000.0, 15000.0, 20000.0], fontsize=16)
            axes[i+3].set_xticklabels(axes[i+3].get_xticks(), fontsize=16)
        else:
            axes[i+3].set_xticklabels(axes[i+3].get_xticks(), fontsize=16)
        # axes[i+3].set_xticklabels(axes[i+3].get_xticks(), fontsize=16)
        axes[i+3].set_yticklabels(categories[i], fontsize=16)
        axes[i+3].set_ylabel('Reduction ratio', fontsize=18)
        axes[i+3].set_xlabel('Memory $(MB)$', fontsize=18)
        axes[i+3].set_title(titles[i], fontsize = 20, y = -0.20)
        axes[i+3].grid(True, alpha=0.5,  axis='x',color='gray', linewidth=1.5, linestyle='--')
        # ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

        # fontsize1 = 12
        # for ii, data in enumerate(data_category1[i]):
        #     axes[i+3].text(ii, data, str(data), ha='center', va='bottom', zorder = 11, fontsize = fontsize1)
        # for ii, data in enumerate(data_category2[i]):
        #     axes[i+3].text(ii+bar_width*1, data, str(data), ha='center', va='bottom', zorder = 11, fontsize = fontsize1)
        # for ii, data in enumerate(data_category3[i]):
        #     axes[i+3].text(ii+bar_width*2, data, str(data), ha='center', va='bottom', zorder = 11, fontsize = fontsize1)
        # for ii, data in enumerate(data_category4[i]):
        #     axes[i+3].text(ii+bar_width*3, data, str(data), ha='center', va='bottom', zorder = 11, fontsize = fontsize1)
        # for ii, data in enumerate(data_category5[i]):
        #     axes[i+3].text(ii+bar_width*4, data, str(data), ha='center', va='bottom', zorder = 11, fontsize = fontsize1)
        # for ii, data in enumerate(data_category6[i]):
        #     axes[i+3].text(ii+bar_width*5, data, str(data), ha='center', va='bottom', zorder = 11, fontsize = fontsize1)
        # for ii, data in enumerate(data_category7[i]):
        #     axes[i+3].text(ii+2, data, str(data), ha='center', va='bottom', zorder = 11, fontsize = fontsize1)


        arrow_length = 0.6  # Proportion of arrow length to the total distance
        arrow_params1 = {'arrowstyle': ']-', 'linewidth': 2.5, 'color': 'black', 'alpha': 0.3}
        arrow_params2 = {'arrowstyle': '->', 'linewidth': 2.5,'color': 'black', 'alpha': 0.3}
        for j in range(len(data_category1[i])):
            # ax.annotate('', xytext=(r1[j], max(data_category7[i])), xy=(r1[j], data_category1[i][j] + 0.05 * (max(data_category7[i])- data_category1[i][j])), arrowprops=arrow_params, annotation_clip=False)
            # ax.annotate('', xytext=(r2[j], max(data_category7[i])), xy=(r2[j], data_category2[i][j] + 0.05 * (max(data_category7[i])- data_category2[i][j])), arrowprops=arrow_params, annotation_clip=False)
            # ax.annotate('', xytext=(r3[j], max(data_category7[i])), xy=(r3[j], data_category3[i][j] + 0.05 * (max(data_category7[i])- data_category3[i][j])), arrowprops = arrow_params, annotation_clip = False)
            # ax.annotate('', xytext=(r4[j], max(data_category7[i])), xy=(r4[j], data_category4[i][j] + 0.05 * (max(data_category7[i])- data_category4[i][j])), arrowprops = arrow_params, annotation_clip = False)
            # ax.annotate('', xytext=(r5[j], max(data_category7[i])), xy=(r5[j], data_category5[i][j] + 0.05 * (max(data_category7[i])- data_category5[i][j])), arrowprops = arrow_params, annotation_clip = False)
            if max(data_category7[i] / data_category6[i][j]) > 100:
                axes[i+3].annotate("" ,
                            xytext=( max(data_category7[i]) * 0.99, r6[j]),
                            xy=( data_category6[i][j] + 0.66 * (max(data_category7[i]) - data_category6[i][j]), r6[j]),
                            arrowprops=arrow_params1, annotation_clip=False)
            else:
                axes[i+3].annotate("" ,
                            xytext=( max(data_category7[i]) * 0.99, r6[j]),
                            xy=( data_category6[i][j] + 0.64 * (max(data_category7[i]) - data_category6[i][j]), r6[j]),
                            arrowprops=arrow_params1, annotation_clip=False)
            axes[i+3].annotate("%.1fx" % max(data_category7[i] / data_category6[i][j]), xytext=(data_category6[i][j] + 0.5 * (max(data_category7[i]) - data_category6[i][j])
            ,r6[j] - 0.03 ),
                        xy=(data_category6[i][j] + 0.04 * (max(data_category7[i]) - data_category6[i][j]), r6[j]),
                        arrowprops=arrow_params2, annotation_clip=False, fontsize=15)  # if i == 0:
        #     ax.set_ylim(0, 35)

    lines, labels = fig.axes[-1].get_legend_handles_labels()
    # Add a legend
    plt.subplots_adjust(top = 0.890, bottom = 0.1500, left = 0.035, right = 0.995, wspace=0.310, hspace=0.2)

    fig.legend(lines, labels, loc = 'upper center', fontsize = 18, ncol = 7)
    # plt.tight_layout()
    # Show the plot
    plt.savefig('nodebatch.pdf', bbox_inches='tight')
    plt.show()


# f1()
# f2()
# f3()
# f4()
# f5()
# f6()
# f7()
f8()


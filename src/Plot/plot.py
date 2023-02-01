import matplotlib.pyplot as plt
import numpy as np


def plot_statistics(csvs, title, labels, type_stat, path_savefig):
    # All files and directories ending with .txt and that don't begin with a dot:
    time = []
    height = []
    for i in range(len(csvs)):
        tmp_time = []
        tmp_height = []
        with open(csvs[i], 'r') as f:
            content = f.readlines()
            content.remove("\n")
            for stat in content:
                if stat != "\n" and stat != "N. Instance,Solution Status,Height,Time,N. Failures\n" and \
                stat != "N. Instance,Solution Status,Height,Time\n":
                    result = stat.split(",")
                    print(result[1])
                    if result[3] == '' or result[1] == "No solution found":
                        tmp_time.append(0)
                    else:
                        tmp_time.append(float(result[3].replace("\n", "")))
                    if result[2] == '' or result[2] == ' ':
                        tmp_height.append(0)
                    else:
                        tmp_height.append(int(result[2]))
        time.append(tmp_time)
        height.append(tmp_height)

    plt.subplots(figsize=(18, 12))
    if len(csvs) >= 3:
        barWidth = 0.25
    else:
        barWidth = 0.4

    if type_stat == "time":
        colors = ['lightsalmon', 'skyblue', 'greenyellow']
        for n_csv in range(len(csvs)):
            plt.bar(np.arange(1, 41)+(barWidth*n_csv), time[n_csv], color=colors[n_csv], width=barWidth, label=labels[n_csv], align='center')
        plt.ylabel('Time in seconds')
        plt.yscale("symlog")
        plt.axis([0, 41, 0, 400])
    else:
        colors = ['chartreuse', 'darkslateblue', 'aqua']
        for n_csv in range(len(csvs)):
            plt.bar(np.arange(1, 41)+(barWidth*n_csv), height[n_csv], color=colors[n_csv], width=barWidth, label=labels[n_csv], align='center')
        plt.ylabel('Height')
        plt.axis([0, 41, 0, 200])

    plt.xlabel('Instances')
    plt.title(title)
    plt.xticks(np.arange(1, 41, 1), range(1, 41))
    plt.legend()
    plt.savefig(path_savefig)



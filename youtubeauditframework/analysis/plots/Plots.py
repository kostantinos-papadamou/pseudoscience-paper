#!/usr/bin/python

import numpy as np
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# Import Plot-related packages
from matplotlib.lines import Line2D
# Set Font style and size
font = {'size': 13}
import matplotlib.pyplot as plt; plt.rc('font', **font)


class Plots(object):
    """
    Class that implements all the methods for producing the plots for the YouTube's Recommendation Algorithm Audit experiments
    """
    @staticmethod
    def plot(plot_items, plot_filename, ylabel, xlabel, x_val_start, x_val_end, x_val_step=1.0,
             ylim_top=None, legend_labels=None, legend_position=None, show_legend=True):
        """
        Method that generates the plot for all types
        :param plot_items:
        :param plot_filename:
        :param ylabel:
        :param xlabel:
        :param x_val_start:
        :param x_val_end:
        :param x_val_step:
        :param ylim_top:
        :param legend_labels:
        :param legend_position:
        :param show_legend:
        :return:
        """
        # Clear plot history
        plt.clf()
        plt.cla()

        """ Plot Configuration """
        # COLORS
        col_blue = "C0"
        col_red = "#A50808"
        col_yellow1 = "#FF9900"
        col_green = "#5F9E6E"
        col_pink = "#F442F1"
        colors = [col_red, col_blue, col_yellow1, col_green, col_pink]

        # LINE STYLES
        linestyle1 = '-'
        linestyle2 = '--'
        linestyle3 = '-'
        linestyle4 = '-.'
        line_styles = [linestyle1, linestyle2, linestyle3, linestyle4, linestyle1, linestyle2, linestyle3, linestyle4]

        # Other plot configuration
        opacity = 0.9
        linewidth = 3

        """ Generate Plot """
        # Create a Sub-plot object
        fig, ax = plt.subplots()

        # Set X-AXIS values according to the number of hops
        x_axis_values = np.arange(x_val_start, x_val_end + 1, x_val_step)
        plt.xticks(x_axis_values)

        # Plot Items (User Profiles)
        for i in range(0, len(plot_items)):
            ax.plot(x_axis_values, plot_items[i], color=colors[i], alpha=opacity, linewidth=linewidth, linestyle=line_styles[i])

        # Set X and Y Axis labels
        plt.xlabel(xlabel)
        plt.ylabel(ylabel, multialignment='center')

        # Add Grids in the plot
        plt.grid(color='#CCCCCC', alpha=0.9, linestyle='--')

        # Set Limits
        plt.ylim(bottom=0)
        if ylim_top is not None:
            plt.ylim(top=ylim_top)

        # Change Font size
        for item in ([ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(17)

        # Add Legend (if requested)
        if show_legend:
            lines, labels = list(), list()
            for i in range(0, len(legend_labels)):
                lines.append(Line2D([0], [0], color=colors[i], linewidth=linewidth, linestyle=line_styles[i]))
                labels.append(legend_labels[i].replace('_', ' '))

            if show_legend:
                ax.legend(lines, labels, loc=legend_position if legend_position is not None else "best", fontsize=16)

        # Set X and Y Axis Limits
        plt.margins(x=0)

        # Save plot as figure
        fig.savefig(filename=plot_filename, format='pdf', bbox_inches='tight')
        plt.show()
        return

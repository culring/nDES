import numpy as np
import matplotlib.pyplot as plt

def get_plot_style(fig_size_multiplier: float = 1.5):
    """ Get style parameters for :mod:`matplotlib`.

    Args:
        fig_size_multiplier: Figure size multiplier.

    Returns:
        Dictionary of style parameters which can be used to alter
        :data:`matplotlib.rcParams` dictionary.
    """
    # golden ratio
    fig_width = 6.750 * fig_size_multiplier
    fig_height = fig_width / 1.618
    params = {
        'axes.labelsize': 10,
        'legend.fontsize': 10,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'text.usetex': False,
        'figure.figsize': [fig_width, fig_height],
        'savefig.bbox': 'tight',
        'savefig.transparent': False
    }

    return params


def set_grid(axis):
    """Set a grid on the axis.

    Args:
        axis: Axis on which the grid will be set.

    Returns:
        Axis with set grid.
    """
    axis.spines['top'].set_visible(False)
    axis.spines['right'].set_visible(False)
    axis.spines['left'].set_visible(False)
    axis.spines['bottom'].set_visible(False)
    axis.get_xaxis().tick_bottom()
    axis.get_yaxis().tick_left()
    axis.tick_params(axis='x', direction='out')
    axis.tick_params(axis='y', direction='out')
    # offset the spines
    for spine in axis.spines.values():
        spine.set_position(('outward', 5))
    # put the grid behind
    axis.set_axisbelow(True)
    axis.grid(color="0.9", linestyle='-', linewidth=1)
    return axis

plt.rcParams.update(get_plot_style(2.5))

if __name__ == "__main__":
    fig, ax = plt.subplots()
    for i in ['4', '8', '16']:
        log = np.load(f'log_{i}k_fixed.npy')
        log = np.clip(log, 0., 3.)
        x = np.linspace(0, 500000, num=len(log))
        ax.plot(x, log, label=(r'$\lambda$ = ' + f'{i}k'))
    ax.legend()
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Log loss')
    ax = set_grid(ax)
    #  plt.show()
    fig.savefig('logloss_lambda_des.png')

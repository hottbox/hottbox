"""
This module is for EDA through interactive visualisation of the core structures:
``TensorCPD``, ``TensorTKD`` etc.

All its functionality should be used inside Jupyter Lab/ Jupyter Notebook.

.. important::

    The API provided by this module is experimental
"""

import numpy as np
import ipywidgets as widgets
import matplotlib.pyplot as plt  # This essentially can be replaced with `plotly` in a future
from scipy import signal  # for generating testing data
from hottbox.core import TensorCPD, TensorTKD  # for type hinting
from IPython.display import display, clear_output


def gen_test_data(plot=False):
    """ Generate factor matrices which components will be easy to differentiate from one another

    Parameters
    ----------
    plot : bool

    Returns
    -------
    fmat : list[np.ndarray]
    core_values : np.ndarray
    """
    t_A = np.linspace(0, 1, 500, endpoint=False).reshape(-1, 1)
    t_B = np.linspace(0, 2, 10, endpoint=False).reshape(-1, 1)
    t_C = np.linspace(-1, 1, 2 * 100, endpoint=False).reshape(-1, 1)
    w_A = np.array([1, 2, 5]).reshape(-1, 1)
    w_B = np.roll(w_A, 1)
    w_C = np.array([0.3, 2, 0.7]).reshape(-1, 1)

    A = np.sin         (2 * np.pi * t_A * w_A.T)
    B = signal.square  (2 * np.pi * t_B * w_B.T)
    C, _, _ = signal.gausspulse(t_C * w_C.T, fc=5, retquad=True, retenv=True)
    fmat = [A, B, C]
    core_values = np.array([1]*A.shape[1])

    if plot:
        for mode, factor in enumerate(fmat):
            print("Mode-{} factor matrix shape = {}".format(mode, factor.shape))
        fig, axis = plt.subplots(nrows=3,
                                 ncols=1,
                                 figsize=(8, 8)
                                 )
        axis[0].plot(t_A, A)
        axis[0].set_title("Factor matrix A")
        axis[1].plot(t_B, B)
        axis[1].set_title("Factor matrix B")
        axis[2].plot(t_C, C)
        axis[2].set_title("Factor matrix C")
        plt.tight_layout()
    return fmat, core_values


def gen_test_tensor_cpd():
    """ Generate ``TensorCPD`` object for testing purposes

    Returns
    -------
    TensorCPD
    """
    return TensorCPD(*gen_test_data())


def _line_plot(ax, data):
    """ Default plotting function for each mode

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes object which is used to illustrate `data`
    data : np.ndarray
        Array of data to be plotted. Shape of such array is ``(N, 1)``
    """
    ax.plot(data)


def _bar_plot(ax, data):
    """ Sample custom plot function """
    ax.bar(x=range(data.shape[0]), height=data)

BANK_OF_PLOTS = {
    "line" : _line_plot,
    "bar" : _bar_plot
}


def _main_plotting_function(tensor_cpd, group, plot_bank):

    n_rows = 1
    n_cols = tensor_cpd.order
    axis_width = 4
    axis_height = 4
    fig, axis = plt.subplots(nrows=n_rows,
                             ncols=n_cols,
                             figsize=(n_cols * axis_width, n_rows * axis_height)
                             )

    for i, fmat in enumerate(tensor_cpd.fmat):
        factor = group[i]
        plot_function = plot_bank[i]
        plot_function(ax=axis[i],
                      data=fmat[:, factor])
        axis[i].set_title("Factor matrix: {}".format(tensor_cpd.mode_names[i]))
    plt.tight_layout()
    return fig



class BaseIPlot(object):

    DEFAULT_PLOT_TYPE = {"line" : _line_plot}

    def __init__(self, tensor_rep):
        """

        Parameters
        ----------
        tensor_rep : {TensorCPD, TensorTKD}
        """
        self.tensor_rep = tensor_rep
        self.out = widgets.Output()
        self.sliders = self._create_fmat_sliders()
        self.dropdown = self._create_fmat_dropdown()
        self.dashboard = widgets.VBox([self.out,
                                       widgets.HBox(self.sliders),
                                       widgets.HBox(self.dropdown)
                                       ])

    def _create_fmat_sliders(self):
        """ Just a dummy slider since this is an interface """
        slider_list = [widgets.IntSlider(min=0, max=0, value=0)]
        return slider_list

    def _create_fmat_dropdown(self):
        # TODO: Needs to be dynamic
        dropdown_params = dict(options=['line', 'bar'],
                               value='line',
                               description='Plot type:',
                               disabled=False
                               )
        dropdown_list = [widgets.Dropdown(**dropdown_params) for _ in self.tensor_rep.fmat]
        return dropdown_list

    def start_interacting(self):
        # Start tracking changes
        [slider.observe(self.general_callback, names="value") for slider in self.sliders]
        [dropdown.observe(self.general_callback, names="value") for dropdown in self.dropdown]
        display(self.dashboard)

    def general_callback(self, change):
        slider_values = [slider.value for slider in self.sliders]
        dropdown_values = [dropdown.value for dropdown in self.dropdown]
        self.update_plot(slider_values=slider_values, dropdown_values=dropdown_values)

    def update_plot(self, slider_values, dropdown_values):
        group = tuple(slider_values)
        plot_bank = {i: BANK_OF_PLOTS[value] for i, value in enumerate(dropdown_values)}
        with self.out:
            fig = _main_plotting_function(tensor_cpd=self.tensor_rep, group=group, plot_bank=plot_bank)
            display(fig)
            clear_output(wait=True)


class PlotTensorCPD(BaseIPlot):
    def __init__(self, tensor_rep):
        super(PlotTensorCPD, self).__init__(tensor_rep=tensor_rep)

    def _create_fmat_sliders(self):
        slider_list = [widgets.IntSlider(min=0, max=(self.tensor_rep.fmat[0].shape[1] - 1))]
        return slider_list

    def _create_fmat_dropdown(self):
        dropdown_list = super(PlotTensorCPD, self)._create_fmat_dropdown()
        return dropdown_list

    def start_interacting(self):
        super(PlotTensorCPD, self).start_interacting()

    def general_callback(self, change):
        super(PlotTensorCPD, self).general_callback(change)

    def update_plot(self, slider_values, dropdown_values):
        super(PlotTensorCPD, self).update_plot(slider_values=slider_values * self.tensor_rep.order,
                                               dropdown_values=dropdown_values
                                               )

class PlotTensorTKD(BaseIPlot):
    def __init__(self, tensor_rep):
        super(PlotTensorTKD, self).__init__(tensor_rep=tensor_rep)

    def _create_fmat_sliders(self):
        slider_list = [widgets.IntSlider(min=0, max=(fmat.shape[1] - 1)) for fmat in self.tensor_rep.fmat]
        return slider_list

    def _create_fmat_dropdown(self):
        dropdown_list = super(PlotTensorTKD, self)._create_fmat_dropdown()
        return dropdown_list

    def start_interacting(self):
        super(PlotTensorTKD, self).start_interacting()

    def general_callback(self, change):
        super(PlotTensorTKD, self).general_callback(change)

    def update_plot(self, slider_values, dropdown_values):
        super(PlotTensorTKD, self).update_plot(slider_values=slider_values,
                                               dropdown_values=dropdown_values
                                               )





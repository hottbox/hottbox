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
from scipy import signal  # for generating testing data and can be removed
from IPython.display import display, clear_output
from ..core import TensorCPD, TensorTKD  # for type hinting

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


_DEFAULT_1D_PLOTS = {
    "line": _line_plot,
    "bar": _bar_plot
}


class BaseComponentPlot(object):

    def __init__(self, tensor_rep):
        """

        Parameters
        ----------
        tensor_rep : {TensorCPD, TensorTKD}
        """
        self.tensor_rep = tensor_rep
        self.available_plots = {**_DEFAULT_1D_PLOTS}
        self.out = widgets.Output()
        self.sliders = self._create_fmat_sliders()
        self.dropdown = self._create_fmat_dropdown()
        self.dashboard = widgets.VBox([self.out,
                                       widgets.HBox(self.sliders),
                                       widgets.HBox(self.dropdown)
                                       ])
        self._start_interacting()

    def _create_fmat_sliders(self):
        """ Just a dummy slider since this is an interface """
        slider_list = [widgets.IntSlider(min=0, max=0, value=0)]
        return slider_list

    def _create_fmat_dropdown(self):
        options_list = list(self.available_plots.keys())
        default_value = "line"  # default_value = options_list[0]
        dropdown_params = dict(options=options_list,
                               value=default_value,
                               description='Plot type:',
                               disabled=False
                               )
        dropdown_list = [widgets.Dropdown(**dropdown_params) for _ in self.tensor_rep.fmat]
        return dropdown_list

    def _start_interacting(self):
        # Start tracking changes
        [slider.observe(self._general_callback, names="value") for slider in self.sliders]
        [dropdown.observe(self._general_callback, names="value") for dropdown in self.dropdown]
        display(self.dashboard)

    def _general_callback(self, change):
        slider_values = [slider.value for slider in self.sliders]
        dropdown_values = [dropdown.value for dropdown in self.dropdown]
        self._update_plot(slider_values=slider_values, dropdown_values=dropdown_values)

    def _update_plot(self, slider_values, dropdown_values):
        group = tuple(slider_values)
        with self.out:
            fig = self._main_plotting_function(group=group)
            display(fig)
            clear_output(wait=True)

    def _main_plotting_function(self, group):
        n_rows = 1
        n_cols = self.tensor_rep.order
        axis_width = 4
        axis_height = 4
        fig, axis = plt.subplots(nrows=n_rows,
                                 ncols=n_cols,
                                 figsize=(n_cols * axis_width, n_rows * axis_height)
                                 )

        for i, fmat in enumerate(self.tensor_rep.fmat):
            factor = group[i]
            plot_function = self.available_plots[self.dropdown[i].value]
            plot_function(ax=axis[i],
                          data=fmat[:, factor])
            axis[i].set_title("Factor matrix: {}".format(self.tensor_rep.mode_names[i]))
        plt.tight_layout()
        return fig

    def _update_figure(self):

        pass

    # TODO: think of an elegant way of adding plot function only to a specific dropdown menu
    def extend_available_plots(self, custom_plots):
        # TODO: this implementation changes the order of available options but the because we eliminate possible duplicates using set which is unordered
        # extend dict with 'label' -> plot_functions
        self.available_plots.update(custom_plots)

        # Update dropdown menus
        new_dropdown_options = list(custom_plots.keys())
        for dropdown in self.dropdown:
            dropdown.options = list({*dropdown.options, *new_dropdown_options})

    # def extend_available_plots(self, custom_plots):
    # # TODO: don't like this implementation but the original order of options should still be preserved
    #     current_dropdown_options = self.dropdown[0].options
    #     new_dropdown_options = [key for key in custom_plots if key not in current_dropdown_options]
    #     self.available_plots.update(custom_plots)
    #     for dropdown in self.dropdown:
    #         dropdown.options = [*dropdown.options, *new_dropdown_options]


class ComponentPlotCPD(BaseComponentPlot):
    def __init__(self, tensor_rep):
        super(ComponentPlotCPD, self).__init__(tensor_rep=tensor_rep)

    def _create_fmat_sliders(self):
        slider_list = [widgets.IntSlider(min=0, max=(self.tensor_rep.fmat[0].shape[1] - 1))]
        return slider_list

    def _create_fmat_dropdown(self):
        dropdown_list = super(ComponentPlotCPD, self)._create_fmat_dropdown()
        return dropdown_list

    def _start_interacting(self):
        super(ComponentPlotCPD, self)._start_interacting()

    def _general_callback(self, change):
        super(ComponentPlotCPD, self)._general_callback(change)

    def _update_plot(self, slider_values, dropdown_values):
        super(ComponentPlotCPD, self)._update_plot(slider_values=slider_values * self.tensor_rep.order,
                                                   dropdown_values=dropdown_values
                                                   )

    def extend_available_plots(self, custom_plots):
        super(ComponentPlotCPD, self).extend_available_plots(custom_plots=custom_plots)


class ComponentPlotTKD(BaseComponentPlot):
    def __init__(self, tensor_rep):
        super(ComponentPlotTKD, self).__init__(tensor_rep=tensor_rep)

    def _create_fmat_sliders(self):
        slider_list = [widgets.IntSlider(min=0, max=(fmat.shape[1] - 1)) for fmat in self.tensor_rep.fmat]
        return slider_list

    def _create_fmat_dropdown(self):
        dropdown_list = super(ComponentPlotTKD, self)._create_fmat_dropdown()
        return dropdown_list

    def _start_interacting(self):
        super(ComponentPlotTKD, self)._start_interacting()

    def _general_callback(self, change):
        super(ComponentPlotTKD, self)._general_callback(change)

    def _update_plot(self, slider_values, dropdown_values):
        super(ComponentPlotTKD, self)._update_plot(slider_values=slider_values,
                                                   dropdown_values=dropdown_values
                                                   )

    def extend_available_plots(self, custom_plots):
        super(ComponentPlotTKD, self).extend_available_plots(custom_plots=custom_plots)

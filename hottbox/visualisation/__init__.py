"""
This module is for EDA through interactive visualisation of the core structures:
``TensorCPD``, ``TensorTKD`` etc.

All its functionality should be used inside Jupyter Lab/ Jupyter Notebook.

.. important::

    The API provided by this module is experimental
"""

import numpy as np
import matplotlib.pyplot as plt  # This essentially can be replaced with `plotly` in a future
from collections import OrderedDict
from scipy import signal  # for generating testing data and can be removed
from ipywidgets import IntSlider, VBox, HBox, Dropdown, Output
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
    """ Represent factor vector as line plot

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes object which is used to illustrate `data`
    data : np.ndarray
        Array of data to be plotted. Shape of such array is ``(N, 1)``
    """
    ax.plot(data)


def _bar_plot(ax, data):
    """ Represent factor vector as bar plot

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes object which is used to illustrate `data`
    data : np.ndarray
        Array of data to be plotted. Shape of such array is ``(N, 1)``
    """
    ax.bar(x=range(data.shape[0]), height=data)


_DEFAULT_1D_PLOTS = OrderedDict([("line", _line_plot),
                                 ("bar", _bar_plot)
                                 ])


class BaseComponentPlot(object):
    """

    Attributes
    ----------
    available_plots : dict[str, callable]
        Ordered dictionary with all available plot functions for the factor vectors
    out : Output
    sliders : list[IntSlider]
    dropdown : list[Dropdown]
    dashboard : VBox

    """

    def __init__(self, tensor_rep):
        """

        Parameters
        ----------
        tensor_rep : {TensorCPD, TensorTKD}
        """
        self.tensor_rep = tensor_rep
        self.available_plots = _DEFAULT_1D_PLOTS.copy()
        self.out = Output()
        self.sliders = self._create_fmat_sliders()
        self.dropdown = self._create_fmat_dropdown()
        self.dashboard = VBox([self.out,
                               HBox(self.sliders),
                               HBox(self.dropdown)
                               ])
        self._start_interacting()

    def _create_fmat_sliders(self):
        """ Create slider widgets for factor vectors

        Returns
        -------
        slider_list : list[IntSlider]
            List of sliders for selecting factor vector to be plotted

        Notes
        -----
            This is a dummy slider
        """
        default_params = dict(min=0,
                              max=0,
                              value=0,
                              continuous_update=False
                              )
        slider_list = [IntSlider(**default_params)]
        return slider_list

    def _create_fmat_dropdown(self):
        """ Create dropdown widgets for each factor matrix

        Returns
        -------
        dropdown_list : list[Dropdown]
            List of dropdown menus for selecting type of plot for the factor matrix
        """
        options_list = list(self.available_plots.keys())
        default_value = options_list[0]  # default_value = "line"
        dropdown_default_params = dict(options=options_list,
                                       value=default_value,
                                       description='Plot type:',
                                       disabled=False
                                       )
        dropdown_list = [Dropdown(**dropdown_default_params) for _ in self.tensor_rep.fmat]
        return dropdown_list

    def _start_interacting(self):
        """ Setup a handler to be called when values of dashboard have been changed
        """
        [slider.observe(self._general_callback, names="value") for slider in self.sliders]
        [dropdown.observe(self._general_callback, names="value") for dropdown in self.dropdown]
        display(self.dashboard)

    def _general_callback(self, change):
        """ A callable that is called when when values of dashboard have been changed

        Notes
        -----
            The signature of this method should not be changed
        """
        slider_values = [slider.value for slider in self.sliders]
        self._update_visualisation(slider_values=slider_values)

    def _update_visualisation(self, slider_values):
        """

        Parameters
        ----------
        slider_values : list[int]
        """
        group = tuple(slider_values)
        with self.out:
            fig = self._plot_factor_vectors(group=group)
            display(fig)
            clear_output(wait=True)

    def _plot_factor_vectors(self, group):
        """

        Parameters
        ----------
        group

        Returns
        -------
        fig : matplotlib.figure.Figure
        """
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

    def extend_available_plots(self, custom_plots, modes):
        """ Add custom plot functions available for representing factor vectors

        This method can be used either for adding new ways of plotting of the
        factor vectors or changing ones that have already been defined.
        Despite of chosen option there are two main steps. First the internal dictionary
        with plots (`self.available_plots`) is updated, then dropdown menus.

        Parameters
        ----------
        custom_plots : dict[str, callable]
            Dictionary with plot functions
            Keys will be displayed in the dropdown menu.
            Values will be used as plotting functions for factor vector
            when the corresponding option from the dropdown menu is selected.
        modes : list[int]
            List of modes for which keys of `custom_plots` will be available in the dropdown.
            If not specified then they will be available for all modes.

        Notes
        -----
            1)  If key from `custom_plots` is already in ``self.available_plots.keys()``,
                then the corresponding value will be updated anyway.
            2)  When the options of dropdown menu are updated, then index of that dropdown
                menu is also getting changed. Looks like it is always set to zero which could
                change the value that is selected in the dropdown menu.
            3)  Signature of plotting functions should contain two variables: `ax` and `data`
                ::

                    def my_line_plot(ax, data):
                        ax.plot(data, 'r+')
        """

        self.available_plots.update(custom_plots)

        if not modes:
            dropdown_update_list = [j for j in range(len(self.dropdown))]
        else:
            dropdown_update_list = modes

        # Update dropdown menus, this will also reset their index
        new_options = list(custom_plots.keys())
        for j in dropdown_update_list:
            old_options = [*self.dropdown[j].options]
            unique_options = {*old_options, *new_options}
            updated_options = [option for option in self.available_plots.keys() if option in unique_options]
            self.dropdown[j].options = tuple(updated_options)


class ComponentPlotCPD(BaseComponentPlot):
    def __init__(self, tensor_rep):
        super(ComponentPlotCPD, self).__init__(tensor_rep=tensor_rep)

    def _create_fmat_sliders(self):
        default_params = dict(min=0,
                              max=self.tensor_rep.fmat[0].shape[1] - 1,
                              value=0,
                              continuous_update=False
                              )
        slider_list = [IntSlider(**default_params)]
        return slider_list

    def _create_fmat_dropdown(self):
        dropdown_list = super(ComponentPlotCPD, self)._create_fmat_dropdown()
        return dropdown_list

    def _start_interacting(self):
        super(ComponentPlotCPD, self)._start_interacting()

    def _general_callback(self, change):
        super(ComponentPlotCPD, self)._general_callback(change)

    def _update_visualisation(self, slider_values):
        super(ComponentPlotCPD, self)._update_visualisation(slider_values=slider_values * self.tensor_rep.order)

    def extend_available_plots(self, custom_plots, modes=()):
        super(ComponentPlotCPD, self).extend_available_plots(custom_plots=custom_plots,
                                                             modes=modes)


class ComponentPlotTKD(BaseComponentPlot):
    def __init__(self, tensor_rep):
        super(ComponentPlotTKD, self).__init__(tensor_rep=tensor_rep)

    def _create_fmat_sliders(self):
        default_params = dict(min=0,
                              value=0,
                              continuous_update=False
                              )
        slider_list = [IntSlider(**default_params, max=(fmat.shape[1] - 1)) for fmat in self.tensor_rep.fmat]
        return slider_list

    def _create_fmat_dropdown(self):
        dropdown_list = super(ComponentPlotTKD, self)._create_fmat_dropdown()
        return dropdown_list

    def _start_interacting(self):
        super(ComponentPlotTKD, self)._start_interacting()

    def _general_callback(self, change):
        super(ComponentPlotTKD, self)._general_callback(change)

    def _update_visualisation(self, slider_values):
        super(ComponentPlotTKD, self)._update_visualisation(slider_values=slider_values)

    def extend_available_plots(self, custom_plots, modes=()):
        super(ComponentPlotTKD, self).extend_available_plots(custom_plots=custom_plots,
                                                             modes=modes)

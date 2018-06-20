"""
Classes that bring interpretability into tensor representations
"""
import itertools


class State(object):
    """ This class describes state of the ``Tensor`` and tracks reshaping modifications

    Attributes
    ----------
    _normal_shape : tuple
        Shape of a tensor object in normal format (without being in unfolded or folded state).
    _mode_order : list[list]
    """

    def __init__(self, normal_shape, mode_order) -> None:
        """

        Parameters
        ----------
        normal_shape : tuple
        mode_order : list[list]
        """
        self._normal_shape = tuple([i for i in normal_shape])
        self._mode_order = mode_order.copy()

    def __eq__(self, other):
        """
        Returns
        -------
        bool
        """
        if isinstance(self, other.__class__):
            return self.__dict__ == other.__dict__
        return False

    def __str__(self):
        self_as_string = "{}(normal_shape={}, mode_order={})".format(self.__class__.__name__,
                                                                     self._normal_shape,
                                                                     self._mode_order)
        return self_as_string

    def __repr__(self):
        return str(self)

    @property
    def normal_shape(self):
        return self._normal_shape

    @property
    def mode_order(self):
        return self._mode_order

    def copy(self):
        """ Produces a copy of itself as a new object

        Returns
        -------
        new_object : Mode
        """
        normal_shape = self.normal_shape
        mode_order = self.mode_order
        new_object = State(normal_shape=normal_shape,
                           mode_order=mode_order)
        return new_object

    def is_normal(self):
        return len(self.normal_shape) == len(self.mode_order)

    def set_mode_order(self, new_mode_order):
        self._mode_order = new_mode_order

    def set_normal_shape(self, new_normal_shape):
        self._normal_shape = new_normal_shape

    def reset_mode_order(self):
        pass

    def unfold(self, mode):
        first_mode = self.mode_order.pop(mode)
        other_modes = list(itertools.chain.from_iterable(self.mode_order))
        self.set_mode_order([first_mode, other_modes])

    def fold(self):
        new_mode_order = [*self.mode_order[0], *self.mode_order[1]]
        new_mode_order.sort()
        self.set_mode_order([[i] for i in new_mode_order])

    def rotate(self):
        pass

    def reset(self):
        pass


class Mode(object):
    """ This class describes mode of the ``Tensor``

    Attributes
    ----------
    _name : str
        Placeholder for the name of the mode
    _index : list[str]
        Placeholder for the list of indices for this mode
    """
    def __init__(self, name) -> None:
        """ Constructor of the ``Mode`` class

        Parameters
        ----------
        name : str
            Name of the mode
        """
        if not isinstance(name, str):
            raise TypeError("Parameter `name` should be a string!")
        self._name = name.strip().replace("_", "-")
        self._index = None

    def __str__(self):
        self_as_string = "{}(name='{}', index={})".format(self.__class__.__name__,
                                                          self.name,
                                                          self.index)
        return self_as_string

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        """
        Returns
        -------
        bool

        Notes
        -----
        Modes are equal when everything is the same.
        """
        if isinstance(self, other.__class__):
            return self.__dict__ == other.__dict__
        return False

    def copy(self):
        """ Produces a copy of itself as a new object

        Returns
        -------
        new_object : Mode
        """
        name = self.name
        index = self.index
        new_object = Mode(name=name)
        new_object.set_index(index=index)
        return new_object

    @property
    def name(self):
        """ Name of the mode

        Returns
        -------
        name : str
        """
        name = self._name
        return name

    @property
    def index(self):
        """ List of indices for the mode

        Returns
        -------
        index : list[str]
        """
        index = self._index
        return index

    def set_name(self, name):
        """ Set new name of the mode

        Parameters
        ----------
        name : str
            New name of the mode

        Returns
        -------
        self : Mode
        """
        if not isinstance(name, str):
            raise TypeError("Parameter `name` should be a string!")
        self._name = name
        return self

    def set_index(self, index):
        """ Set new list of indices for the mode

        Parameters
        ----------
        index : list

        Returns
        -------
        self : Mode
        """
        if index is not None:
            if not isinstance(index, list):
                raise TypeError("Parameter `index` should be a list!")

        self._index = index
        return self

    def reset_index(self):
        """ Drop list of indices for the mode

        Returns
        -------
        self : Mode
        """
        self._index = None
        return self

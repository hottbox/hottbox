"""
Classes that bring interpretability into tensor representations
"""
import itertools


class State(object):
    """ This class describes state of the ``Tensor`` and tracks reshaping modifications

    Attributes
    ----------
    _normal_shape : tuple
        Shape of a `Tensor` object in normal format (without being in unfolded or folded state).
    _transformations : list[tuple]
        List of transformations applied to ``Tensor``. Starts with the default `(None, [[0], [1], ..., [N-1]])`.
        Each transformation is defined by the type of reshaping ("T", "K") and order of the modes (list of lists).
        Each transformation is represented as a tuple of length 2.
        The first element specifies type of reshaping ("T", "K").
        The second element specifies order of the tensor modes in form of list of lists.
    """

    def __init__(self, normal_shape, mode_order=None, reshaping=None) -> None:
        """

        Parameters
        ----------
        normal_shape : tuple
        mode_order : list[list]
        reshaping : str
            Type of reshaping: {"T", "K"}
        """
        normal_mode_order_ = [[i] for i in range(len(normal_shape))]
        self._transformations = [("Init", normal_mode_order_)]
        self._normal_shape = tuple([i for i in normal_shape])

        if mode_order is not None and reshaping is not None:
            transformation = (reshaping, mode_order.copy())
            self._transformations.append(transformation)

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
        self_as_string = "{}(normal_shape={}, mode_order={}, reshaping='{}')".format(self.__class__.__name__,
                                                                                     self.normal_shape,
                                                                                     self.mode_order,
                                                                                     self.reshaping)
        return self_as_string

    def __repr__(self):
        return str(self)

    @property
    def normal_shape(self):
        """ Shape of a `Tensor` object in normal state (without being in unfolded or folded).

        Returns
        -------
        tuple
        """
        return self._normal_shape

    def change_normal_shape(self, new_normal_shape):
        """ Change shape of a `Tensor` object in normal format dut to mode-n product or contraction

        Parameters
        ----------
        new_normal_shape : tuple
        """
        self._normal_shape = new_normal_shape

    @property
    def normal_mode_order(self):
        """ Order of the modes of a `Tensor` object in normal state (without being in unfolded or folded).

        Returns
        -------
        list[list]
            This list takes form ``[[0], [1], ..., [N-1]]`` where N is the
            order of the `Tensor` in the normal state
        """
        normal_form = self._transformations[0]
        return normal_form[1]

    @property
    def last_transformation(self):
        """ Last transformation applied to `Tensor` object """
        return self._transformations[-1]

    @property
    def mode_order(self):
        """ Order of the modes of a `Tensor` object after the last transformation

        Returns
        -------
        list[list]
        """
        return self.last_transformation[1].copy()

    @property
    def reshaping(self):
        """ Type of the last reshaping applied to a `Tensor` object

        Returns
        -------
        str
        """
        return self.last_transformation[0]

    def is_normal(self):
        """ Checks if a `Tensor` object in normal state

        Returns
        -------
        bool
        """
        return self.mode_order == self.normal_mode_order

    def add_transformation(self, new_mode_order, rtype):
        """ Add transformation applied to `Tensor` object

        Parameters
        ----------
        new_mode_order : list[list]
        rtype : str
        """
        transformation = (rtype, new_mode_order)
        self._transformations.append(transformation)

    def remove_transformation(self):
        """ Remove the last transformation applied to `Tensor` object """
        if len(self._transformations) > 1:
            del self._transformations[-1]

    def unfold(self, mode, rtype):
        """ Registers an unfolding operation applied to a `Tensor` object

        Parameters
        ----------
        mode : int
        rtype : str
        """
        current_mode_order = self.mode_order
        first_mode = current_mode_order.pop(mode)
        other_modes = list(itertools.chain.from_iterable(current_mode_order))
        new_mode_order = [first_mode, other_modes]
        self.add_transformation(new_mode_order=new_mode_order, rtype=rtype)

    def fold(self):
        """ Register a folding operation applied to a `Tensor` object (reverts unfolding) """
        self.remove_transformation()

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
        self._name = name.strip().replace("_", "-")
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

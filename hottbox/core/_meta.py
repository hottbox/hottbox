"""
Classes that bring interpretability into tensor representations
"""
import itertools


class State(object):
    """ This class describes state of the ``Tensor`` and tracks reshaping modifications

    Attributes
    ----------
    _normal_shape : tuple
        Shape of a ``Tensor`` object in normal format (without being in unfolded or folded state).
    _transformations : list[tuple]
        List of transformations applied to ``Tensor``.
        Each transformation is represented as a tuple of length 2 ``(rtype, mode_order)``.
        This list of transformations starts with the default ``('Init', ([0], [1], ..., [N-1]))``.
        Each transformation is specifies:
            1) the reshaping type as a string from {"T", "K"}.
            2) order of the tensor modes modes as tuple of lists with integers.
    """

    def __init__(self, normal_shape, rtype=None, mode_order=None) -> None:
        """ Constructor for the ``State`` for the ``Tensor`` object

        Parameters
        ----------
        normal_shape : tuple
            Shape of a ``Tensor`` object in normal format (without being in unfolded or folded state).
        rtype : str
            Type of reshaping: {"T", "K"}.\n
            Optional parameter used during creation of the ``Tensor`` object which has data in modified form.
            Should be used with ``mode_order``
        mode_order : tuple[list]
            Optional parameter used during creation of the ``Tensor`` object which has data in modified form.
            Should be used with ``rtype``
        """
        normal_mode_order_ = tuple([i] for i in range(len(normal_shape)))
        self._transformations = [("Init", normal_mode_order_)]
        self._normal_shape = tuple(i for i in normal_shape)

        if mode_order is not None and rtype is not None:
            transformation = (rtype, mode_order)
            self._transformations.append(transformation)

    def __eq__(self, other):
        """
        Parameters
        ----------
        other : State

        Returns
        -------
        bool
        """
        equal = False
        if isinstance(self, other.__class__):
            normal_shape_equal = self.normal_shape == other.normal_shape
            number_transforms_equal = len(self._transformations) == len(other._transformations)
            transforms_equal = False
            if number_transforms_equal:
                transforms_equal = all([self._transformations[i] == t for i, t in enumerate(other._transformations)])
            equal = normal_shape_equal and number_transforms_equal and transforms_equal
        return equal

    def __str__(self):
        self_as_string = "{}(normal_shape={}, rtype='{}', mode_order={})".format(self.__class__.__name__,
                                                                                 self.normal_shape,
                                                                                 self.rtype,
                                                                                 self.mode_order)
        return self_as_string

    def __repr__(self):
        return str(self)

    @property
    def normal_shape(self):
        """ Shape of a ``Tensor`` object in normal state (without being in unfolded or folded).

        Returns
        -------
        tuple
        """
        return self._normal_shape

    @property
    def transformations(self):
        """ List of transformations applied to ``Tensor``.

        Returns
        -------
        list[tuple]
        """
        return self._transformations

    @property
    def normal_mode_order(self):
        """ Order of the modes of a ``Tensor`` object in normal state (without being in unfolded or folded).

        Returns
        -------
        tuple[list]
            Takes form ``([0], [1], ..., [N-1])`` where `N` is the
            order of the `Tensor` in the normal state
        """
        normal_form = self._transformations[0]
        return normal_form[1]

    @property
    def last_transformation(self):
        """ Last transformation applied to ``Tensor`` object """
        return self._transformations[-1]

    @property
    def mode_order(self):
        """ Order of the modes of a ``Tensor`` object after the last transformation

        Returns
        -------
        tuple[list]
        """
        return self.last_transformation[1]

    @property
    def rtype(self):
        """ Type of the last reshaping applied to a ``Tensor`` object

        Returns
        -------
        str
        """
        return self.last_transformation[0]

    def is_normal(self):
        """ Checks if a ``Tensor`` object in normal state

        Returns
        -------
        bool
        """
        return self.mode_order == self.normal_mode_order

    def change_normal_shape(self, normal_shape):
        """ Change shape of a ``Tensor`` object in normal format due to mode-n product or contraction

        Parameters
        ----------
        normal_shape : tuple

        Returns
        -------
        self
        """
        self._normal_shape = normal_shape
        return self

    def add_transformation(self, rtype, mode_order):
        """ Add transformation applied to ``Tensor`` object

        Parameters
        ----------
        rtype : str
            Reshaping type as a string from {"T", "K"}.
        mode_order : tuple[list]

        Returns
        -------
        self
        """
        transformation = (rtype, mode_order)
        self._transformations.append(transformation)
        return self

    def remove_transformation(self):
        """ Remove the last transformation applied to ``Tensor`` object

        Returns
        -------
        self
        """
        if len(self._transformations) > 1:
            del self._transformations[-1]
        return self

    def unfold(self, mode, rtype):
        """ Register an unfolding operation applied to a ``Tensor`` object

        Parameters
        ----------
        mode : int
        rtype : str
            Reshaping type as a string from {"T", "K"}.

        Returns
        -------
        self
        """
        current_mode_order = [i for i in self.mode_order]
        first_mode = current_mode_order.pop(mode)
        other_modes = list(itertools.chain.from_iterable(current_mode_order))
        new_mode_order = (first_mode, other_modes)
        self.add_transformation(rtype=rtype, mode_order=new_mode_order)
        return self

    def vectorise(self, rtype):
        """ Register a vectorisation operation applied to a ``Tensor`` object

        Parameters
        ----------
        rtype : str
            Reshaping type as a string from {"T", "K"}.

        Returns
        -------
        self
        """
        new_mode_order = tuple([list(itertools.chain.from_iterable(self.mode_order))])
        self.add_transformation(rtype=rtype, mode_order=new_mode_order)
        return self

    def fold(self):
        """ Register a folding operation applied to a ``Tensor`` object (reverts unfolding)

        Returns
        -------
        self
        """
        self.remove_transformation()
        return self

    def reset(self):
        """ Reset ``State`` to the initial, unmodified form

        Returns
        -------
        self
        """
        self._transformations = self._transformations[:1]
        return self


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

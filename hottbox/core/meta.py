"""
Classes that bring interoperability into tensor representations
"""


class Mode(object):
    """ This class describe mode of the Tensor

    Attributes
    ----------
    _name : str
    _index : list[str]
    """
    def __init__(self, name) -> None:
        """ Constructor of the ``Mode`` class

        Parameters
        ----------
        name : str
        """
        if not isinstance(name, str):
            raise TypeError("name should be a string")
        self._name = name
        self._index = None

    def __str__(self):
        self_as_string = "{}(name=['{}'], index=[{}])".format(self.__class__.__name__,
                                                              self.name,
                                                              self.index)
        return self_as_string

    def __repr__(self):
        return str(self)

    def copy(self):
        """ Produces a copy of itself as a new object

        Returns
        -------
        new_object : Mode
        """
        name = self.name
        index = self.index
        new_object = Mode(name=name)
        new_object.set_index(new_index=index)
        return new_object

    @property
    def name(self):
        """

        Returns
        -------
        name : str
        """
        name = self._name
        return name

    @property
    def index(self):
        """

        Returns
        -------
        index : list
        """
        index = self._index
        return index

    def set_name(self, new_name):
        """

        Parameters
        ----------
        new_name : str

        """
        self._name = new_name

    def set_index(self, new_index):
        """

        Parameters
        ----------
        new_index : list

        """
        self._index = new_index

    def reset_index(self):
        self._index = None

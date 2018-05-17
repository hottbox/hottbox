import pytest
from .._meta import Mode


class TestMode:
    """ Tests for ``Mode`` class """
    def test_init(self):
        """ Tests for constructor of the ``Mode`` class """
        true_name = "tensor-mode"
        mode = Mode(name=true_name)
        assert mode.index is None
        assert mode.name == true_name

        name = "tensor_mode"
        true_name = "tensor-mode"
        mode = Mode(name=name)
        assert mode.name == true_name

        name = "tensor mode"
        true_name = "tensor mode"
        mode = Mode(name=name)
        assert mode.name == true_name

        name = " tensor mode "
        true_name = "tensor mode"
        mode = Mode(name=name)
        assert mode.name == true_name

        mode = Mode(name="tensor-mode")
        mode.set_index(["index"])
        true_mode_as_string = "Mode(name=['tensor-mode'], index=[['index']])"
        assert repr(mode) == true_mode_as_string

        with pytest.raises(TypeError):
            Mode(name=5)

    def test_copy(self):
        orig_name = "mode"
        orig_index = ["index"]
        mode = Mode(name=orig_name)
        mode.set_index(orig_index)

        mode_copy = mode.copy()
        assert mode_copy is not mode
        assert mode_copy.name == orig_name
        assert mode_copy.index == orig_index

        # test for change to the copy not affecting orig object and vice versa
        mode_copy.set_name("new-name")
        mode_copy.set_index(["new-index"])
        assert mode_copy.name != mode.name
        assert mode_copy.index != mode.index

    def test_set_name(self):
        name = "tensor-mode"
        true_new_name = "mode"
        mode = Mode(name=name)
        mode.set_name("mode")
        assert mode.name == true_new_name

        with pytest.raises(TypeError):
            mode.set_name(5)

    def test_set_index(self):
        """ Tests for `set_index` method """
        name = "name"
        index = ["index"]
        mode = Mode(name=name)
        mode.set_index(index=index)
        assert mode.index == index

        mode.set_index(index=None)
        assert mode.index is None

        with pytest.raises(TypeError):
            mode.set_index(index=tuple(index))

    def test_reset_index(self):
        """ Tests for `reset_index` method """
        name = "name"
        index = ["index"]
        mode = Mode(name=name)
        mode.set_index(index=index)
        assert mode.index == index
        mode.reset_index()
        assert mode.index is None

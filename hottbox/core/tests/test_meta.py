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
        true_mode_as_string = "Mode(name='tensor-mode', index=['index'])"
        assert repr(mode) == true_mode_as_string

        assert mode != name

        with pytest.raises(TypeError):
            Mode(name=5)

    def test_copy(self):
        """ Tests for `copy` method """
        orig_name = "mode"
        orig_index = ["index"]
        mode = Mode(name=orig_name).set_index(index=orig_index)

        mode_copy = mode.copy()
        assert mode_copy == mode
        assert mode_copy is not mode

        # test for change to the copy not affecting orig object and vice versa
        mode_copy.set_name("new-name").set_index(["new-index"])
        assert mode_copy != mode

    def test_set_name(self):
        """ Tests for `set_name` method """
        name = "tensor-mode"
        true_new_name = "mode"
        mode = Mode(name=name).set_name("mode")
        assert mode.name == true_new_name

        with pytest.raises(TypeError):
            mode.set_name(5)

    def test_set_index(self):
        """ Tests for `set_index` method """
        name = "name"
        true_new_index = ["index"]
        mode = Mode(name=name).set_index(index=["index"])
        assert mode.index == true_new_index

        mode.set_index(index=None)
        assert mode.index is None

        with pytest.raises(TypeError):
            mode.set_index(index=tuple(true_new_index))

    def test_reset_index(self):
        """ Tests for `reset_index` method """
        name = "name"
        index = ["index"]
        mode = Mode(name=name).set_index(index=index)
        assert mode.index == index
        mode.reset_index()
        assert mode.index is None

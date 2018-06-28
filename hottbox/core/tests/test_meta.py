import pytest
import sys
import io
from .._meta import Mode, State


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


class TestState:
    """ Tests for ``State`` class """
    def test_init(self):
        """ Tests for constructor of the ``State`` class """
        normal_shape = (2, 3, 4)
        true_normal_shape = normal_shape
        true_default_rtype = "Init"
        true_default_mode_order = ([0], [1], [2])
        true_default_transformations = (true_default_rtype, true_default_mode_order)
        true_custom_rtype = "T"
        true_custom_mode_order = ([0], [1, 2])
        true_custom_transformation = (true_custom_rtype, true_custom_mode_order)

        state1 = State(normal_shape=normal_shape)
        assert state1.is_normal()
        assert state1.normal_shape == true_normal_shape
        assert state1.normal_shape is not true_normal_shape
        assert len(state1.transformations) == 1
        assert state1.last_transformation == true_default_transformations
        assert state1.mode_order == true_default_mode_order
        assert state1.rtype == true_default_rtype

        state2 = State(normal_shape=normal_shape, rtype=true_custom_rtype, mode_order=true_custom_mode_order)
        assert len(state2.transformations) == 2
        assert state2.last_transformation == true_custom_transformation
        assert state2.mode_order == true_custom_mode_order
        assert state2.rtype == true_custom_rtype

        mode = Mode(name="mode")
        assert state1 != mode

        # Tests for __str__ and __repr__
        captured_output = io.StringIO()  # Create StringIO object
        sys.stdout = captured_output     # and redirect stdout.
        print(repr(state2))
        assert captured_output.getvalue() != ''  # to check that something was actually printed

    def test_change_normal_shape(self):
        """ Tests for `change_normal_shape` method """
        normal_shape = (2, 2, 2)
        new_normal_shape = (2, 3, 4)
        true_number_transforms = 1
        state = State(normal_shape=normal_shape)
        state.change_normal_shape(normal_shape=new_normal_shape)
        assert state.is_normal()
        assert state.normal_shape == new_normal_shape
        assert len(state.transformations) == true_number_transforms

    def test_add_transformation(self):
        """ Tests for `add_transformation` method """
        normal_shape = (2, 2, 2)
        normal_shape_1 = (2, 3, 4)
        normal_shape_2 = (3, 4, 5)
        rtype_1 = "T"
        rtype_2 = "K"
        mode_order_1 = ([0], [1, 2])
        mode_order_2 = ([1], [0, 2])

        state1 = State(normal_shape=normal_shape_1)
        state2 = State(normal_shape=normal_shape_2)
        assert state1 != state2

        state1 = State(normal_shape=normal_shape).add_transformation(rtype=rtype_1, mode_order=mode_order_1)
        state2 = State(normal_shape=normal_shape).add_transformation(rtype=rtype_1, mode_order=mode_order_2)
        assert state1 != state2

        state1 = State(normal_shape=normal_shape).add_transformation(rtype=rtype_1, mode_order=mode_order_1)
        state2 = State(normal_shape=normal_shape).add_transformation(rtype=rtype_2, mode_order=mode_order_1)
        assert state1 != state2

        state1 = State(normal_shape=normal_shape, rtype=rtype_1, mode_order=mode_order_1)
        state2 = State(normal_shape=normal_shape).add_transformation(rtype=rtype_1, mode_order=mode_order_1)
        assert state1 == state2
        assert state1.last_transformation == (rtype_1, mode_order_1)

        state1.add_transformation(rtype=rtype_2, mode_order=mode_order_2)
        assert state1 != state2
        assert state1.last_transformation == (rtype_2, mode_order_2)

        state1.reset()
        normal_shape = state1.normal_shape
        assert state1 == State(normal_shape=normal_shape)

    def test_unfold_vectorise_fold(self):
        """ Tests for `unfold`, `vectorise` and `fold` methods """
        normal_shape = (2, 3, 4)
        default_rtype = "Init"
        unfold_rtype = "T"
        vec_rtype = "K"
        default_mode_order = ([0], [1], [2])
        unfold_mode_order = [([0], [1, 2]),
                             ([1], [0, 2]),
                             ([2], [0, 1])
                             ]
        vec_mode_order = ([0, 1, 2],)

        state = State(normal_shape=normal_shape)
        assert state.mode_order == default_mode_order
        assert state.rtype == default_rtype
        state.fold()
        assert state.mode_order == default_mode_order
        assert state.rtype == default_rtype

        for i, true_mode_order in enumerate(unfold_mode_order):
            state.unfold(mode=i, rtype=unfold_rtype)
            assert state.mode_order == true_mode_order
            assert state.rtype == unfold_rtype
            state.fold()
            assert state.mode_order == default_mode_order
            assert state.rtype == default_rtype

        state.vectorise(rtype=vec_rtype)
        assert state.mode_order == vec_mode_order
        assert state.rtype == vec_rtype
        state.fold()
        assert state.mode_order == default_mode_order
        assert state.rtype == default_rtype

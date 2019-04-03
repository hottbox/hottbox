Before pull request
===================

If it isn’t tested, it’s broken
-------------------------------

Unit tests are crucial part of this library.
Before submitting a pull request, make sure that all new functions and classes come with tests that completely
cover new features. We use tools provided by the ``pytest`` package to verify that all statements perform as expected.

First, you will need to install some additional software: ::

    $ pip install -e '.[tests]'

To run test, simply execute inside the directory(ies) with your changes: ::

    $ pytest -v --cov . --cov-branch --cov-report term-missing

Here is a `youtube video <https://www.youtube.com/watch?v=ixqeebhUa-w&t=831s>`_ that could help you
if you have never been dealing with this side of software development.



Document your code
------------------

Documentation is a crucial part of this library.
All functions and classes should come with useful docstrings.
For these, we use the **numpy style** docstrings.
Detailed guidelines can be found `here1 <https://numpydoc.readthedocs.io/en/latest/format.html>`_
and `here2 <https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt>`_.
Also you can have a look at the `examples <http://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_numpy.html>`_.
Can be useful to read introduction to structuring documentation of your module `using Sphinx <https://pythonhosted.org/an_example_pypi_project/sphinx.html>`_.



It is advised to check you documentation after you have edited the source code.
You can edit the documentation using any text editor and then generate the HTML output by typing ``make html``
from the docs/ directory::

    $ cd docs
    $ make clean
    $ make html


.. important::
   Building the documentation requires the ``sphinx``, ``numpydoc`` and ``sphinx_rtd_theme``::

      $ pip install -e '.[docs]'



Docker validation
-----------------

Optionally, you can use `Docker <https://youtube.com/watch?v=JprTjTViaEA>`_ and corresponding
utilities from the ``Makefile``, in order to ensure that there are no local dependecies.
Specificaitons for the Docker image can be found in corresponding under ``docker/`` directory::

    # Create docker image with installation of local state of 'hottbox'
    $ make dev-image  # Size 850MB; built on top of conda/miniconda3

    # Start docker container of the image from previous step
    $ make dev-container

    # Inside docker container; Should finish without errors
    $ make test-cov     # performs unit tests with coverage
    $ make html         # builds documentation


.. note::
   .. image:: ../_static/docs_comparison.png
      :scale: 35 %
      :align: center
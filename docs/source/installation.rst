==================
Installing HOTTBOX
==================

There are two options available:

1.  Install ``hottbox`` as it is from `pypi.org <https://pypi.org/project/hottbox/>`_
    by executing: ::

        $ pip install hottbox

2.  Alternatively, you can clone the source code which you can find on our `GitHub repository <https://github.com/hottbox/hottbox>`_
    and install ``hottbox`` in editable mode:
    ::

        $ git clone https://github.com/hottbox/hottbox.git

        $ cd hottbox

        $ pip install -e .

    This will allow you to modify the source code in the way it will suit your needs. Additionally, you will be
    on top of the latest changes and will be able to start using new stable features which are located on
    `develop <https://github.com/hottbox/hottbox/tree/develop>`_ branch until the official release.



.. note::
    If you wish to contribute, please, read the development guide and follow instructions therein


Running tests
-------------

If you have chosen the second installation option, it is advisable to run tests in order to make sure that your
current version of ``hottbox`` is stable. First, you will need to install ``pytest`` and ``pytest-cov`` packages: ::

    $ pip install pytest pytest-cov

To run test, simply execute inside the main directory: ::

    $ pytest -v --cov hottbox



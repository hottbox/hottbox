HOTTBOX: Higher Order Tensors ToolBOX
=====================================

|Travis|_ |Appveyor|_ |Coveralls|_ |PyPi|_ |Binder|_

.. |Travis| image:: https://img.shields.io/travis/hottbox/hottbox/master.svg?label=TravisCI
.. _Travis: https://travis-ci.org/hottbox/hottbox/

.. |Appveyor| image:: https://ci.appveyor.com/api/projects/status/sh2rk41gpn26h7a7/branch/master?svg=true
.. _Appveyor: https://ci.appveyor.com/project/IlyaKisil/hottbox-6jq6a

.. |Coveralls| image:: https://coveralls.io/repos/github/hottbox/hottbox/badge.svg?branch=master
.. _Coveralls: https://coveralls.io/github/hottbox/hottbox?branch=master

.. |PyPi| image:: https://badge.fury.io/py/hottbox.svg
.. _PyPi: https://badge.fury.io/py/hottbox

.. |Binder| image:: https://mybinder.org/badge.svg
.. _Binder: https://mybinder.org/v2/gh/hottbox/hottbox-tutorials/master?urlpath=lab/

Welcome to the toolbox for tensor decompositions, statistical analysis, visualisation, feature extraction,
regression and non-linear classification of multi-dimensional data. Not sure you need this toolbox? Give it
a try on `mybinder.org <https://mybinder.org/v2/gh/hottbox/hottbox-tutorials/master?urlpath=lab/>`_ without installation.



Installing HOTTBOX
------------------

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
    `develop <https://github.com/hottbox/hottbox/tree/develop>`_ branch until the official release. The list
    of provisional changes for the next release (and the CI status) can be also be found on develop branch
    in `CHANGELOG <https://github.com/hottbox/hottbox/blob/develop/CHANGELOG.md>`_ file.



Running tests
-------------

``hottbox`` is under active development, therefore, if you have chosen the second installation
option, it is advisable to run tests in order to make sure that your
current version of ``hottbox`` is stable. First, you will need to install ``pytest`` and ``pytest-cov`` packages: ::

    $ pip install -e '[.tests]'

To run tests, simply execute inside the main directory: ::

    $ pytest -v --cov hottbox



HOTTBOX tutorials
-----------------

Please check out `our repository <https://github.com/hottbox/hottbox-tutorials>`_ with tutorials on ``hottbox`` api
and theoretical background on multi-linear algebra and tensor decompositions.


Development
-----------
We welcome new contributors of all experience levels. Detailed guidelines can be found on
`our web site <https://hottbox.github.io/stable/development_guide/index.html>`_.

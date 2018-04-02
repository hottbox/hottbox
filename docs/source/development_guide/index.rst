=================
Development guide
=================

We appreciate any contributions to the ``hottbox``.
If you have improvements, features or patches, please send us your pull requests (PR)!
You can find specific instructions on how to issue a PR on github `here <https://help.github.com/articles/about-pull-requests/>`_.
Feel free to open an issue if you find a bug or directly come chat with us on our `gitter channel <http://www.example.com/>`_.


How to contribute
-----------------
The preferred way to contribute to ``hottbox`` is to fork the main repository on GitHub,
work on your feature and then submit a PR. The outline of this process should be as follows:

1. Fork the `project repository <https://github.com/hottbox/hottbox>`_ by clicking on the **Fork** button near the top of the page.
   This creates a copy of the code under your account on the GitHub server.
   For more details on how to fork a repository see `this guide <https://help.github.com/articles/fork-a-repo/>`_.

2. Clone this copy to your local disk
   ::

      git clone git@github.com:YourGitHubLogin/hottbox.git
      cd hottbox

3. Most likely you would need (want) to install ``hottbox`` package into you environment in editable mode
   ::

      pip install --editable .

   This basically builds the extension in place and creates a link to the development directory

3. Next, create a branch to hold your changes. It is a good practice not to work on the ``master`` branch
   ::

      git checkout -b my-feature

4. Work on this copy, on your computer, using Git to do the version control. In order to record your changes in Git, do
   ::

      git add modified_files
      git commit -m "**Concise but meaningful description**"

   When you’re done with changes, then push them to GitHub with
   ::

      git push -u origin my-feature

5. Finally, follow `these <https://help.github.com/articles/creating-a-pull-request-from-a-fork/>`_ instructions to create a pull request from your fork.

.. note::
   When you decide which branch you'd like to merge your changes into (step 4 of PR guide above),
   you should almost always select ``develop`` branch from the *base branch* drop-down menu.


Advised git branching model
---------------------------

We are following git branching model `described here  <http://nvie.com/posts/a-successful-git-branching-model/>`_
with some amendments. Key points:

1. ``master`` branch: only stable version of the library
2. ``develop`` branch: only stable features that will be included in the next release
3. ``feature`` branches: for developing features, create a tag when stable
4. ``release`` branches: for shaping up the documentation and fixing minor bugs
5. ``hotfix`` branches: for fixing critical bugs that exist on the ``master`` branch

For more details and conventions have a look at

.. toctree::
   :maxdepth: 2

   git_branching_model


Coding guidelines
-----------------
Can be found on the official `website <http://scikit-learn.org/dev/developers/contributing.html#coding-guidelines>`_
of ``scikit-learn``


APIs of hottbox objects
-----------------------
We should also adhere ``scikit-learn`` `type of API <http://scikit-learn.org/dev/developers/contributing.html#apis-of-scikit-learn-objects>`_ for objects of our classes


Documentation
-------------

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

    cd docs
    make clean
    make html


.. note::
   Building the documentation requires the ``sphinx``, ``numpydoc`` and ``sphinx_rtd_theme``::

      pip install sphinx numpydoc sphinx_rtd_theme
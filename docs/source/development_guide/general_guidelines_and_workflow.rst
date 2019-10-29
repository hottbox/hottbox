General guidelines and advices
==============================

.. contents:: Table of Contents
    :local:
    :depth: 2

Coding guidelines
-----------------
Can be found on the official `website <http://scikit-learn.org/dev/developers/contributing.html#coding-guidelines>`_
of ``scikit-learn``



APIs of hottbox objects
-----------------------
We should also adhere ``scikit-learn`` `type of API <http://scikit-learn.org/dev/developers/contributing.html#apis-of-scikit-learn-objects>`_
for objects of our classes



Advised git branching model
---------------------------

We are following git branching model `described here  <http://nvie.com/posts/a-successful-git-branching-model/>`_
with some amendments. Key points:

1. ``master`` branch: only stable version of the library
2. ``develop`` branch: only stable features that will be included in the next release
3. ``feature`` branches: for developing features, create a tag when stable
4. ``release`` branches: for shaping up the documentation and fixing minor bugs
5. ``hotfix`` branches: for fixing critical bugs that exist on the ``master`` branch

More details and conventions for each type of branches:

.. toctree::
   :maxdepth: 2

   git_branching_model



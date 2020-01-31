===================
Git branching model
===================

.. contents:: Table of Contents
    :local:
    :depth: 2

Master branch
=============
We consider ``origin/master`` to be the main branch where the source code of HEAD
always reflects a production-ready state.

.. note::
    Don't forget to add tag whenever you merge to the master branch.

Develop branch
==============
We consider ``origin/develop`` to be the main branch where the source code of HEAD
always reflects a state with the latest delivered development changes for the next release.

When the source code in the ``develop`` branch reaches a stable point and is ready to be released,
all of the changes should be merged back into ``master`` somehow and then **tagged** with a release number.

Supporting branches
===================

Feature branches
----------------

Used to develop new features for the upcoming or a distant future release. Therefore you should branch off from
``develop`` (or its child) and merge back to the same branch which you have branched off

.. code-block:: bash

    git checkout -b myfeature develop   # Create and switch to a new branch

    git checkout develop                # Switch back after you have implemented feature

    ...

    git merge --no-ff myfeature         # Merge your changes.

    git push origin develop

.. important::
    In general, the step of merging a feature branch to the development branch should be done through **pull request**

Release branches
----------------

Used for preparation of a new production release. This includes minor bug fixes and shaping up the documentation.
The key moment to branch off a new release branch from ``develop`` is when develop (almost) reflects the desired
state of the new release. This means that only targeted features for the upcomming release had been merged to
the ``develop`` branch

.. code-block:: bash

    git checkout -b release-1.2 develop

    ...

    git commit -a -m "Last minute changes to the version number 1.2"

    git checkout master

    git merge --no-ff release-1.2 # Include summary of new features to the commit message

    git tag -a 1.2

    git checkout develop

    git merge --no-ff release-1.2 # Include summary of last minute fixes to the commit message

.. important::
    Release branches must be merged back to ``master`` **and** ``develop``

Hotfix branches
---------------

Very much like release branches. However, should only be used when there is the necessity to act immediately
upon an undesired state (critical bug) of a live production version.

.. important::
    When a ``release`` branch currently exists, the hotfix changes need to be merged into that
    ``release`` branch, instead of ``develop``.

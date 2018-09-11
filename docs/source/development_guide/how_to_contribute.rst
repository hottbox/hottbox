How to contribute
=================

Initial setup for your development
----------------------------------

The preferred way to contribute to ``hottbox`` is to fork the main repository on GitHub,
work on your feature and then submit a PR. The outline of this process should be as follows:

1. Fork the `project repository <https://github.com/hottbox/hottbox>`_ by clicking on the **Fork** button near the top of the page.
   This creates a copy of the code under your account on the GitHub server.
   For more details on how to fork a repository see `this guide <https://help.github.com/articles/fork-a-repo/>`_.

2. Clone this copy to your local disk
   ::

      $ git clone git@github.com:__YourGitHubLogin__/hottbox.git
      $ cd hottbox

3. Install ``hottbox`` package (in editable mode) and additional development tools
   ::

      $ pip install -e '.[tests, docs]'

   This basically builds the extension in place and creates a link to the development directory

4. It is a good practice not to work on the ``master`` branch as it should contain only production
   ready state of the code. In case of ``hottbox``, this corresponds to the version available on
   `pypi.org <https://pypi.org/project/hottbox/>`_. Therefore, all development is taking place on
   ``develop`` branch and for each new feature we advise to create a new branch that stems from ``develop``.
   ::

      $ git checkout develop
      $ git checkout -b my-feature

5. Work on this copy, on your computer, using Git to do the version control. In order to record your changes in Git, do
   ::

      $ git add modified_files
      $ git commit -m "**Concise but meaningful description**"

   When you’re done with changes, then push them to GitHub with
   ::

      $ git push -u origin my-feature

6. Finally, follow `these <https://help.github.com/articles/creating-a-pull-request-from-a-fork/>`_ instructions to create a pull request from your fork.

.. note::
   When you decide which branch you'd like to merge your changes into (step 4 of PR guide above),
   you should almost always select ``develop`` branch from the *base branch* drop-down menu.



Getting up to date with the main project
----------------------------------------

1. Add a remote that points to the the Git repo (`main hottbox repo <https://github.com/hottbox/hottbox>`_) from which you want to get the latest changes.
   Conventionally it is referred to as **upstream**: ::

      $ git remote add upstream https://github.com/hottbox/hottbox.git

   Verify new remote by executing this command: ::

      $ git remote -v

      ...
      upstream  https://github.com/hottbox/hottbox.git (fetch)  # <--- should see this
      upstream  https://github.com/hottbox/hottbox.git (push)   # <--- should see this
      ...

2. Next, you need to bring the latest commits from this upstream and merge them
   in order to be in sync with the upstream: ::

      # bring the latest commits
      $ git fetch upstream

      # sync your master and develop branches
      $ git checkout master

      $ git merge upstream/master

      $ git checkout develop

      $ git merge upstream/develop

3. Finally, your can update your own GitHub repo: ::

      $ git push

More details can be found in GitHub's official document on `syncing a fork <https://help.github.com/articles/syncing-a-fork/>`_
and in relevant `discussion <https://stackoverflow.com/questions/7244321/how-do-i-update-a-github-forked-repository>`_
on stackoverflow.
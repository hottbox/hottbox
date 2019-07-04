#!/usr/bin/env bash

set -e

### Auto deploy documentation
# Implementation is taken from https://github.com/scikit-image/scikit-image/blob/master/tools/travis/deploy_docs.sh with minor modifications.
# See https://help.github.com/articles/creating-an-access-token-for-command-line-use/ for how to generate a token
# See https://docs.travis-ci.com/user/encryption-keys/ for how to generate a secure variable on Travis
# See https://github.com/Syntaf/travis-sphinx for a standalone script for automated building and deploying of sphinx docs via travis-ci


if [[ $TRAVIS_PULL_REQUEST == false && $TRAVIS_BRANCH == "develop" && $DEPLOY_DOCS == 1 ]]; then
	echo "-- Pushing docs --"

    (
    git config --global user.email "hottbox.developers@gmail.com"
    git config --global user.name "Travis Bot"

	# Install the dependencies for making documentation
	pip install sphinx guzzle_sphinx_theme numpydoc m2r

	# cd to the doc folder and build the doc
	(cd docs && make html)

    # Clone repo with documentation and update its content
    git clone --quiet https://github.com/hottbox/hottbox.github.io docs_build
    cd docs_build
    git rm -r develop/*
    if [ ! -d develop ]; then
        mkdir develop
    fi
    cp -r ../docs/build/html/* develop

    # In order for contributors to take advantage of their own Travis CI and successfully pass the build
    # there is a check whether 'GH_TOKEN' had been defined in a list of ENV variables in travis settings.
    # Setting it in 'https://travis-ci.org/__USER__/hottbox/settings' would still result errored or failed
    # CI build because their 'GH_TOKEN' would not have write write permissions to 'https://github.com/hottbox/hottbox.github.io'
    # and added to list of ENV variables in travis settings for 'https://travis-ci.org/hottbox/hottbox/settings'.
    if [ ! -z "$GH_TOKEN" ];then
        git add develop
        git commit -m "Travis auto-update (hottbox:develop)"
        git push --force --quiet "https://${GH_TOKEN}@github.com/hottbox/hottbox.github.io" > /dev/null 2>&1
    fi
    )
else
    echo "===================================================="
    echo "Will only push docs if: "
    echo "1) Travis CI is triggered from hottbox:develop."
    echo "2) In case of CI triggered by Pull Request, only if it has been approved."
    echo "3) If \$DEPLOY_DOCS == 1 (specified in '.travis.yml')."
fi

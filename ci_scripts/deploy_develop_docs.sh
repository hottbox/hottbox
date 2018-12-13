#!/usr/bin/env bash

# Implementation is taken from https://github.com/scikit-image/scikit-image/blob/master/tools/travis/deploy_docs.sh with minor modifications.
# See https://help.github.com/articles/creating-an-access-token-for-command-line-use/ for how to generate a token
# See https://docs.travis-ci.com/user/encryption-keys/ for how to generate a secure variable on Travis

if [[ $TRAVIS_PULL_REQUEST == false && $TRAVIS_BRANCH == "develop" && $DEPLOY_DOCS == 1 ]]; then
	echo "-- Pushing docs --"

    git config --global user.email "hottbox.developers@gmail.com"
    git config --global user.name "Travis Bot"

	# Installed the dependencies for making documentation
	pip install sphinx sphinx_rtd_theme numpydoc

	# cd to the doc folder and build the doc
	cd doc
	make html
	cd ..

    git clone --quiet https://github.com/hottbox/hottbox.github.io doc_build
    cd doc_build
    git rm -r develop/*
    cp -r ../doc/_build/html/* develop/

    git add develop
    git commit -m "Travis auto-update (hottbox:develop)"
    git push --force --quiet "https://${GH_TOKEN}@github.com/hottbox/hottbox.github.io" > /dev/null 2>&1

else
    echo "-- Will only push docs from develop and if \$DEPLOY_DOCS == 1 --"
fi

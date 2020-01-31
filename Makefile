#################################################################################
# This Makefile is self-documented
# Use ## before target to provide a description
#################################################################################
.DEFAULT_GOAL := help

SHELL = /bin/bash


#################################################################################
# INSTALL/SETUP COMMANDS                                                        #
#################################################################################
.PHONY: install, install-dev

## Install this package
install:
	pipenv install .

## Install this package with packages required for development
install-dev:
	pipenv install -e '.[tests, docs]'



#################################################################################
# CHECK CODE QUALITY COMMANDS                                                   #
#################################################################################
.PHONY: test test-cov check-code check-all

## Run pytest
test:
	pipenv run pytest -v hottbox

## Run pytest with coverage
test-cov:
	pipenv run pytest -v --cov hottbox --cov-branch --cov-report term-missing

## Check code quality
check-code:
	pipenv run bandit -r hottbox -c bandit.yml -f html -o .reports/hottbox-bandit.html

## Run full check of hottbox
check-all: test-cov check-code



#################################################################################
# CLEAN COMMANDS                                                                #
#################################################################################
.PHONY: clean clean-build clean-test clean-all

## Remove Python file artifacts
clean:
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

## Remove build artifacts
clean-build:
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -f {} +

## Remove test and coverage artifacts
clean-test:
	find . -name '*.coverage' -exec rm -f {} +
	find . -name '*.coverage.*' -exec rm -f {} +
	rm -fr .tox/
	rm -fr htmlcov/
	rm -fr .reports/
	rm -fr .pytest_cache

## Remove all build, test, coverage and Python artifacts
clean-all: clean clean-build clean-test



#################################################################################
# DOCKER RELATED COMMANDS                                                       #
#################################################################################
.PHONY: base-image dev-image dev-container

## Create base docker image
base-image:
	@printf "\n\n"
	@printf "======================================\n"
	@printf "===                                ===\n"
	@printf "===   Creating base docker image   ===\n"
	@printf "===                                ===\n"
	@printf "======================================\n\n"
	docker build -t hottbox-dev-base \
				 -f docker/hottbox-dev-base.dockerfile \
				 .

## Create docker image with hottbox installed in development mode
dev-image: base-image
	@printf "\n\n"
	@printf "=============================================\n"
	@printf "===                                       ===\n"
	@printf "===   Creating development docker image   ===\n"
	@printf "===                                       ===\n"
	@printf "=============================================\n\n"
	docker build -t hottbox-dev \
				 -f docker/hottbox-dev.dockerfile \
				 .

## Run docker container with hottbox installed in development mode
dev-container:
	docker run -it --hostname=localhost --rm hottbox-dev



#################################################################################
# For self-documenting of Makefile: use '##' before target to provide a description
#
# References:
# https://marmelab.com/blog/2016/02/29/auto-documented-makefile.html
# https://github.com/drivendata/cookiecutter-data-science/blob/master/%7B%7B%20cookiecutter.repo_name%20%7D%7D/Makefile
#
# sed script explained:
# /^##/:
# 	* save line in hold space
# 	* purge line
# 	* Loop:
# 		* append newline + line to hold space
# 		* go to next line
# 		* if line starts with doc comment, strip comment character off and loop
# 	* remove target prerequisites
# 	* append hold space (+ newline) to line
# 	* replace newline plus comments by `---`
# 	* print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http://stackoverflow.com/a/11799865/1968>
#
#################################################################################
.PHONY: help

## Show this message
help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
	@echo
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=25 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')

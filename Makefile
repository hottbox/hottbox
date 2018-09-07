SHELL = /bin/bash


.PHONY: test test-cov code-check check-all
test:
	pytest -v hottbox

test-cov:
	pytest -v --cov hottbox --cov-branch --cov-report term-missing


code-check:
	bandit -r hottbox -c bandit.yml -f html -o .reports/hottbox-bandit.html

check-all: test-cov code-check



.PHONY: install
install:
	pip install -e .



.PHONY: test-image base-image dev-image dev-container
base-image:
	docker build -t hottbox-dev-base \
				 -f docker/hottbox-dev-base \
				 .

dev-image:
	docker build -t hottbox-dev \
				 -f docker/hottbox-dev \
				 .

dev-container:
	docker run -it --hostname=localhost --rm hottbox-dev

test-image:
	docker build -t test .
#	docker run -it --rm test

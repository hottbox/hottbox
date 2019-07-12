SHELL = /bin/bash


.PHONY: test test-cov code-check check-all
test:
	pytest -v hottbox

test-cov:
	pytest -v --cov hottbox --cov-branch --cov-report term-missing

code-check:
	bandit -r hottbox -c bandit.yml -f html -o .reports/hottbox-bandit.html

check-all: test-cov code-check


.PHONY: html
html:
	rm -rf docs/source/api/generated
	$(MAKE) -C docs html



.PHONY: install, install-dev
install:
	pip install .

install-dev:
	pip install -e '.[tests, docs]'



.PHONY: test-image base-image dev-image dev-container
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

dev-container:
	docker run -it --hostname=localhost --rm hottbox-dev

test-image:
	docker build -t test .
	docker run -it --rm test

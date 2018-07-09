
test:
	pytest -v hottbox

test-cov:
	pytest -v --cov hottbox --cov-branch --cov-report term-missing

install:
	pip install -e .

code-check:
	bandit -r hottbox -c bandit.yml -f html -o .reports/hottbox-bandit.html

check-all: test-cov code-check
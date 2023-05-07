init:
	pip install -U pip
	pip install -r requirements-dev.txt
	pre-commit install

format:
	black . --line-length 110
	isort . --skip-gitignore --profile black

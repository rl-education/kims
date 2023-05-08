init:
	pip install -U pip
	pip install -r requirements-dev.txt
	pre-commit install

format:
	black src --line-length 110
	isort src --skip-gitignore --profile black

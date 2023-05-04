install-pdm:
	@echo "Install pdm";\
	if [ `command -v pip` ];\
		then pip install pdm;\
	else\
		curl -sSL https://raw.githubusercontent.com/pdm-project/pdm/main/install-pdm.py | python3 -;\
	fi;

all: init format lint

check: format lint

init:
	@echo "Construct RL Development Environment";\
	if [ -z $(VIRTUAL_ENV) ]; then echo Warning, Virtual Environment is required; fi;\
	if [ -z `command -v pdm` ];\
		then make install-pdm;\
	fi;\
	pip install -U pip
	pdm install --prod

init-dev:
	@echo "Construct RL Development Environment";\
	if [ -z $(VIRTUAL_ENV) ]; then echo Warning, Virtual Environment is required; fi;\
	if [ -z `command -v pdm` ];\
		then make install-pdm;\
	fi;\
	pip install -U pip
	pdm install
	pdm run pre-commit install

format:
	pdm run black .

lint:
	pdm run pyright src
	pdm run ruff src --fix

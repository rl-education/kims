ISAAC_PYTHON:=~/.local/share/ov/pkg/isaac_sim-2022.1.1/python.sh

init:
	$(ISAAC_PYTHON) -m pip install -U pip
	$(ISAAC_PYTHON) -m pip install -r requirements.txt
	$(ISAAC_PYTHON) -m pip install -e .

GYM_VERSION := $(shell pip show gym | grep '^Version' | cut -f2 -d':')
# if gym version is not 0.21.0, reinstall the requirements
run-part-1:
	if [ "$(GYM_VERSION)" != "0.21.0" ]; then \
		$(ISAAC_PYTHON) -m pip uninstall -y -r requirements-part-2.txt; \
		$(ISAAC_PYTHON) -m pip install -r requirements-part-1.txt; \
	fi
	$(ISAAC_PYTHON) rl_simulation_class/train_single.py --config_path ${CURDIR}/config

# if gym version is not 0.23.0, reinstall the requirements
run-part-2:
	if [ "$(GYM_VERSION)" != "0.23.0" ]; then \
		$(ISAAC_PYTHON) -m pip uninstall -y -r requirements-part-1.txt; \
		$(ISAAC_PYTHON) -m pip install -r requirements-part-2.txt; \
	fi
	$(ISAAC_PYTHON) rl_simulation_class/train_multi.py --config_path ${CURDIR}/config

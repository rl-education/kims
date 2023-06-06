ISAAC_PYTHON_LOCAL:=~/.local/share/ov/pkg/isaac_sim-2022.1.1/python.sh
ISAAC_PYTHON_REMOTE:=/dli/task/isaac-sim/python_fix.sh

RUN_REMOTE:=1

ifeq ($(RUN_REMOTE),1)
	ISAAC_PYTHON:=$(ISAAC_PYTHON_REMOTE)
else
	ISAAC_PYTHON:=$(ISAAC_PYTHON_LOCAL)
endif

init:
ifeq ($(RUN_REMOTE),1)
	@echo "Running on remote"
	@echo "Copying python_fix.sh"
	mv ./python_fix.sh $(ISAAC_PYTHON_REMOTE)
	chmod +x $(ISAAC_PYTHON_REMOTE)
else
	@echo "Running on local"
endif
	$(ISAAC_PYTHON) -m pip install -U pip
	$(ISAAC_PYTHON) -m pip install -r requirements.txt
	$(ISAAC_PYTHON) -m pip install -e .
	$(ISAAC_PYTHON) rl_simulation_class/first_load.py

get-remote-ip:
	$(ISAAC_PYTHON) rl_simulation_class/get_remote_ip.py

GYM_VERSION := $(shell pip show gym | grep '^Version' | cut -f2 -d':')
# if gym version is not 0.21.0, reinstall the requirements
run-part-1:
# kill running instances of isaac sim
	kill -9 $(pgrep -f "kit") | true
	if [ "$(GYM_VERSION)" != "0.21.0" ]; then \
		$(ISAAC_PYTHON) -m pip uninstall -y -r requirements-part-2.txt; \
		$(ISAAC_PYTHON) -m pip install -r requirements-part-1.txt; \
	fi
	$(ISAAC_PYTHON) rl_simulation_class/train_single.py --config_path ${CURDIR}/config

# if gym version is not 0.23.0, reinstall the requirements
run-part-2:
# kill running instances of isaac sim
	kill -9 $(pgrep -f "kit") | true
	if [ "$(GYM_VERSION)" != "0.23.0" ]; then \
		$(ISAAC_PYTHON) -m pip uninstall -y -r requirements-part-1.txt; \
		$(ISAAC_PYTHON) -m pip install -r requirements-part-2.txt; \
	fi
	$(ISAAC_PYTHON) rl_simulation_class/train_multi.py --config_path ${CURDIR}/config

run-websocket:
# kill running instances of isaac sim
	kill -9 $(pgrep -f "kit") | true
	/dli/task/isaac-sim/isaac-sim.headless.websocket.sh --allow-root

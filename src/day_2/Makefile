init:
	make -C ../../ init
	pr
	pip install wheel == 0.38.4
	pip install setuptools==57.1.0
	pip install -r requirements.txt

format:
	make -C ../../ format

tensorboard:
	tensorboard --logdir=runs &

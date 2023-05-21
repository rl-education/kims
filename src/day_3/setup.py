from __future__ import absolute_import, annotations, division, print_function

from setuptools import find_packages, setup

setup(
    name="rl_simulation_class",
    author="Yuri Rocha",
    version="0.0.1",
    description="Class for creating a RL task in Isaac Sim.",
    keywords=["robotics", "rl"],
    include_package_data=True,
    install_requires=[
        "gym == 0.21.0",
    ],
    packages=find_packages("."),
    classifiers=["Programming Language :: Python :: 3.8"],
)

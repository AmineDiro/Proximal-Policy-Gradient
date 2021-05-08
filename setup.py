from os.path import join, dirname, realpath
from setuptools import setup


setup(
    name='ppo',
    py_modules=['spinup'],
    version='0.1',
    install_requires=[
        'gym[atari,box2d,classic_control]~=0.15.3',
        'ipython',
        'joblib',
        'matplotlib',
        'mpi4py',
        'numpy',
        'pandas',
        'pytest',
        'psutil',
        'scipy',
        'seaborn',
        'torch',
        'tqdm'
    ],
    description="PPO implementation",
    author="DIRHOUSSI Amine",
)

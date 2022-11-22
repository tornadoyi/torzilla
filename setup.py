from setuptools import setup, find_packages

NAME = 'torzilla'

setup(
    name=NAME,
    version='v0.1',
    description="Torzzila is RL trainer based on PyTorch",
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    install_requires=[
        # base
        'gym',
        
        # datas
    ],
    entry_points={
        # 'console_scripts': [
        #     'share = share.cli:main',
        # ],
    },
)
from setuptools import setup, find_packages

setup(name='T2IScoreScore',
    version='0.1',
    description='A package for evaluating metrics for text-to-image model faithfulness.',
    url='https://github.com/michaelsaxon/T2IScoreScore',
    author='Michael Saxon',
    author_email='saxon@ucsb.edu',
    packages=find_packages(where='src'),
    python_requires='>=3.8',
    install_requires=[
        'torch'
        'transformers[torch]'
        'pandas',
        'tqdm',
        'scipy',
        'seaborn',
    ],
    zip_safe=False
)

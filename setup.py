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

'''
    extras_require={
        # 'cogview' : ['git+https://github.com/Sleepychord/Image-Local-Attention.git'],
        # https://github.com/THUDM/CogView2/tree/main
        'figures' : ['seaborn'],
        'altdiffusion' : ['sentencepiece'],
        'openai' : ['openai', 'backoff'],
        'craiyon' : ['dalle-mini', 'jaxlib==0.3.25', 'vqgan-jax @ git+https://github.com/patil-suraj/vqgan-jax.git'],
        'creator' : ['babelnet', 'translators']
    },
    entry_points={
        'console_scripts': [
            'cccl-build-benchmark=cococrola.create.benchmark:main',
            'cccl-evaluate=cococrola.evaluate.evaluate_model:main'
        ]
    },
    '''
from setuptools import setup, find_packages

setup(
    name='biofuse',
    version='0.2.0',
    description='Multi-modal Fusion Framework for Biomedical Foundation Models',
    author='Mirza Hossain',
    author_email='mnh3@st-andrews.ac.uk',
    url='https://github.com/mnhcorp/biofuse',
    packages=find_packages(),
    install_requires=[
        'torch>=2.0.0',
        'torchvision>=0.15.0',
        'numpy>=1.20.0',
        'scikit-learn>=1.0.0',
        'xgboost>=1.5.0',
        'catboost>=1.0.0',
        'medmnist>=2.0.0',
        'Pillow>=9.0.0',
        'tqdm>=4.60.0',
        'click>=8.0.0',
        'pyyaml>=5.4.0',
        'huggingface_hub>=0.20.0',
        'transformers>=4.30.0',
        'timm>=0.9.0',
    ],
    extras_require={
        'dev': [
            'pytest>=7.0.0',
            'black>=22.0.0',
            'ruff>=0.0.250',
            'mypy>=0.950',
        ],
    },
    entry_points={
        'console_scripts': [
            'biofuse=biofuse.cli:main',
        ],
    },
    python_requires='>=3.8',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Medical Science Apps.',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
)
from setuptools import setup

setup(
    name='fast-weight-ae',
    version='0.1.0',    
    description='Autoencoder using GNN and fast-weight dot product',
    url='https://github.com/smorad/fast-weight-autoencoder',
    author='Steven Morad',
    author_email='stevenmorad@gmail.com',
    packages=['fast_weight_ae'],
    install_requires=[
		'torch_geometric',
		'torch',                     
    ],
    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'Operating System :: POSIX :: Linux',        
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],
)

from setuptools import setup, find_packages


setup(
    name='mask_detector',
    version='0.1',
    description='Model for mask detector.',
    author='HTD_AI_Team',
    packages=find_packages(include=['mask_detector', 'mask_detector.*']),
    install_requires=[
        'numpy==1.20.3',
        'pytorch-lightning==1.3.4',
        'torch==1.8.1',
        'torchvision==0.9.1',
        'tqdm==4.41.0',
        'pandas==1.2.4',
        'scikit-learn==0.24.2',
        'pillow==8.2.0',
        'opencv-python==4.5.2.54'
    ]
)

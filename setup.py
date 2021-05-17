from setuptools import setup, find_packages

setup(
    name='mask_model',
    version='0.1',
    description='Model for mask detector.',
    author='HTD_AI_Team',
    packages=find_packages(include=['mask_detector', 'mask_detector.*']),
    install_requires=[
        'numpy==1.19.5',
        'pytorch-lightning',
        'torch',
        'torchvision',
        'tqdm==4.56.0',
        'pandas==1.2.0',
        'scikit-learn==0.24.0',
        'opencv-python==4.5.1.48',
        'pillow==8.1.0'
    ]
)
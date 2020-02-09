from setuptools import setup, find_packages

setup(name='crfseg',
      version='0.1',
      description='PyTorch implementation of conditional random field for multiclass semantic segmenation.',
      url='http://github.com/migonch/crfseg',
      author='Mikhail Goncharov, Stanislav Shimovolos',
      author_email='goncharov.myu@phystech.edu',
      license='MIT',
      packages=find_packages(),
      install_requires=[
          'numpy',
          'torch',
          'torchvision',
      ],
      zip_safe=False)

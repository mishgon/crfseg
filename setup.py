import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='crfseg',
    version='0.1.3',
    description='PyTorch implementation of conditional random field for multiclass semantic segmenation.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='http://github.com/migonch/crfseg',
    author='Mikhail Goncharov, Stanislav Shimovolos',
    author_email='goncharov.myu@phystech.edu',
    license='MIT',
    packages=setuptools.find_packages(),
    python_requires='>=3.6',
    install_requires=[
      'numpy',
      'torch',
      'torchvision',
    ],
    zip_safe=False
)


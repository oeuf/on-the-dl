from setuptools import setup, find_packages
setup(
    name = "otdl",
    version = "0.0.1",
    packages = find_packages(),
    author='Nick Gustafson',
    author_email='njgustafson@gmail.com',
    url='https://github.com/oeuf/on-the-dl',
    description='Python Client for Accessing the Udacity API',
    long_description=open('README.md').read(),
    install_requires=[
        'numpy',
        'tensorflow',
    ],
)

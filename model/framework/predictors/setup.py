from setuptools import setup, find_namespace_packages


setup(
    name='predictors',
    version='1',
    packages=find_namespace_packages(include=[
        'rlm.*',
        'features.*'
        'utilities.*'
    ]),
    zip_safe=False
)
from setuptools import find_packages, setup

from HPAImageClassifier import __version__

setup(
    name="HPAImageClassifier",
    version=__version__,
    description="HPA Image Classification",
    author=["Parisa Mojiri"],
    author_email="parisa.mojiri@gmail.com",
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    long_description=open("README.md").read(),
)

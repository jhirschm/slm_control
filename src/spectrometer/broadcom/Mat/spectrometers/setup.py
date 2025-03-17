from setuptools import setup, find_packages

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="spectrometers",
    version="0.0.1",
    description="A module to inteface with various spectrometers",
    license="None",
    long_description=long_description,
    author="Man Foo",
    author_email="foomail@foo.example",
    url="http://www.foopackage.example/",
    packages=[
        "mplwidget",
        "spectrometers",
        "spectrometers.devices",
        "spectrometers.ui",
        "usb",
        "rgbdriverkit",
    ],
    install_requires=["seabreeze", "numpy", "matplotlib", "pyqt5"],
)

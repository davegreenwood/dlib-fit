"""Basic seup for entry points."""
from setuptools import setup

setup(name='dlib-fit',
      version='0.1',
      description='Fit dlib 68 to images or video.',
      author='Dave Greenwood',
      py_modules=["dlib_fit"],
      zip_safe=False,
      entry_points={'console_scripts': [
          "dlib-fit=dlib_fit:track",
          "vid2png=dlib_fit:extract"]}
      )

from setuptools import setup
import gtfblib

setup(name='gtfblib',
      version='0.2.0',
      description='A selection of Gammatone Filterbanks',
      url='http://github.com/jthiem/gtfblib',
      author='Joachim Thiemann',
      author_email='Joachim.Thiemann@gmail.com',
      license='CC-BY 3.0',
      packages=['gtfblib'],
      test_suite = 'nose2.collector.collector',
      install_requires=[
          'scipy>=0.18.0,<2.0.0'
      ],
      zip_safe=False)

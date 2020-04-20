from setuptools import setup

setup(name = 'qulati',
      version = '1.0',
      description = 'Quantifying Uncertainty in Local Activation Time Interpolation',
      url = 'http://github.com/samcoveney/quLATi',
      author = 'Sam Coveney',
      author_email = 'coveney.sam@gmail.com',
      license = 'GPL-3.0+',
      packages = ['qulati'],
      install_requires = [
          'numpy',
          'scipy',
          'matplotlib',
          'future',
          'numba',
          'trimesh'
      ],
      zip_safe = False)

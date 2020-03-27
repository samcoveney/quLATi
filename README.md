# quLATi

This package is for *Quantifying Uncertainty for Local Activation Time Interpolation*.

It implements *Gaussian Process Manifold Interpolation (GPMI)* for doing Gaussian process regression on a manifold represented by a triangle mesh.

Since the code is focussed on GP regression on atrial meshes, the eigenfunction calculation routine makes use of specialized algorithms for mesh extension, subdivision, and gradient calculations. These could easily be modified for other use cases.


## How to install

It is probably best to set up a specialized anaconda environment for using quLATi:

We set up such an environment in this order:

```bash

conda create --name pylat

source activate pylat

conda install -c conda-forge numpy scipy matplotlib numba trimesh

```

The dependencies can be installed without anaconda using pip.

Then run the following command from the top directory of quLATi:

```bash

python setup.py install

```

## How to use

Coming soon!


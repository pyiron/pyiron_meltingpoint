# Interface Method
[![codecov](https://codecov.io/gh/pyiron/pyiron_meltingpoint/graph/badge.svg?token=J2QZCMSUD4)](https://codecov.io/gh/pyiron/pyiron_meltingpoint)
[![Pipeline](https://github.com/pyiron/pyiron_meltingpoint/actions/workflows/pipeline.yml/badge.svg)](https://github.com/pyiron/pyiron_meltingpoint/actions/workflows/pipeline.yml)

The *Melting* Jupyter notebook allows the fully automated computation of melting points of unary crystals for arbitrary interatomic potentials that are compatible with the molecular dynamics engine [LAMMPS](https://lammps.sandia.gov). It is based on the interface method where the evolution of the solid and the liquid phase are monitored as function of temperature. The only mandatory input parameters required are the chemical element and the interatomic potential file. The melting point protocol itself is implemented in the standalone [`interfacemethod`](https://github.com/pyiron/pyiron_meltingpoint) Python package located in the `src` directory of this repository, the notebook in the `scripts` directory is merely a thin driver around it.

# Different Versions
The melting point simulation protocol is continously improved based on the feedback from different users. This [repository](https://github.com/pyiron/pyiron_meltingpoint) always includes the latest version, older versions are available as tagged releases.

* **Version 2.0** - Modern modular software stack: the [pyiron](http://pyiron.org)- and `snakemake`-based workflow is replaced by [`interfacemethod`](https://github.com/pyiron/pyiron_meltingpoint), a lightweight standalone package. LAMMPS is now driven directly via [lammpsparser](https://github.com/pyiron/lammpsparser), structure analysis uses [structuretoolkit](https://github.com/pyiron/structuretoolkit) and parallel as well as HPC queue execution is handled by [executorlib](https://github.com/pyiron/executorlib). The `interfacemethod` package is published on [PyPI](https://pypi.org) and [conda-forge](https://conda-forge.org), so the melting point protocol can now be installed and used without pyiron.
* **Version 1.3** - Update dependencies to use `pyiron_atomistics` rather than `pyiron`.
* **Version 1.2** - Fix numpy Version to 1.19.5.
* **Version 1.1** - Adds support for diamond structures. In addition [pyscal](https://pyscal.org) is used for structure analysis, python 3.9 support is added as well as support for Mac OS X. On windows it is recommended to use the linux subsystem for windows.
* **Version 1.0** - This version was originally published in Computational Materials Science. It uses [ovito](https://www.ovito.org) for structure analysis and supports bcc, fcc and hcp structures.

# Installation
`interfacemethod` and all its dependencies - including LAMMPS - are available on conda-forge and are pinned in the [`environment.yml`](environment.yml) file of this repository. There is no separate installation step for the package itself, installing the environment is sufficient to run the notebooks in the `scripts` folder directly.

Either create a new, dedicated environment (the `environment.yml` file does not set a name, so pick one with `-n`):
```
conda env create -n pyiron_meltingpoint -f environment.yml
conda activate pyiron_meltingpoint
```
or install the dependencies into an environment you already have activated:
```
conda env update -f environment.yml
```
`interfacemethod` is also available on PyPI (`pip install interfacemethod`), but LAMMPS itself and a number of the other dependencies are not pure Python packages, so installing via conda/`environment.yml` is the recommended and tested way to get a working setup.

This repository is developed and continuously tested primarily on Linux, in addition the unit tests are also tested on macOS; Windows users are recommended to use the Linux subsystem for Windows (WSL).

## Run the Jupyter Notebook
The melting point protocol is executed from [*script.ipynb*](scripts/script.ipynb). Copy the notebook to the directory you want to run the calculation in - typically next to an `input.json` file and the corresponding potential file, for example one of the [examples](examples) discussed below - and start Jupyter there.

The first cells of the notebook define the calculation:
* `project_path` - working directory the individual LAMMPS calculations are executed in.
* `input_file` / `output_file` - the `input.json` file the calculation is loaded from (if it exists) and the `output.json` file the final melting point prediction as well as the intermediate results are written to.
* `lmp_command` - the command used to call LAMMPS, by default `lmp -in lmp.in`. To parallelise the individual LAMMPS calls with MPI prefix it accordingly, e.g. `mpirun -n 4 lmp -in lmp.in`.
* `max_workers` - the number of LAMMPS calculations [executorlib](https://github.com/pyiron/executorlib) is allowed to run concurrently on the local machine.
* `project_parameter` - the numerical settings of the melting point protocol (number of atoms, run lengths, convergence criteria, ...) together with default values which can be overwritten via the `input.json` file.

The content of the `input.json` file is:
```json
{
    "config": [
        "pair_style eam/alloy \n",
        "pair_coeff * * Fe-C-Bec07.eam Fe C\n"
    ],
    "filename": "Fe-C-Bec07.eam",
    "species": ["Fe", "C"],
    "element": "Fe"
}
```
All remaining cells of the notebook - marked with *"From here on the notebook is automated - no change required"* - execute the interface method fully automatically: an initial melting temperature bracket is estimated, a solid-liquid interface is built and iterated on, and the iteration is repeated until the convergence criterion is reached or a maximum number of iterations is exceeded. The `output.json` file is updated after every step, so an interrupted calculation can be continued by simply rerunning the notebook in the same directory. The last cells of the notebook plot the convergence of the predicted melting temperature over the iterations, no separate analysis notebook is required any more.

## Examples
The [examples](examples) directory contains ready to use `input.json` files and interatomic potentials for a bcc ([Fe](examples/bccFe)), fcc ([Al](examples/fccAl)), hcp ([Mg](examples/hcpMg)) and diamond ([Si](examples/diaSi)) structure, together with a short `README.md` describing the source of each potential. To run one of them, copy `scripts/script.ipynb` into the corresponding example folder and start Jupyter there, so the relative paths in `input.json` resolve correctly:
```
cp scripts/script.ipynb examples/bccFe/
cd examples/bccFe
jupyter notebook script.ipynb
```

# FAQ
## How to run in parallel?
A single melting point calculation takes 50-100 CPU hours, so it makes a lot of sense to run the code in parallel. There are two independent levers:
* `max_workers` in the notebook controls how many LAMMPS calculations [executorlib](https://github.com/pyiron/executorlib)'s `SingleNodeExecutor` is allowed to dispatch concurrently, e.g. the individual strain points of the interface method are embarrassingly parallel and are submitted at once.
* `lmp_command` controls how each individual LAMMPS call itself is parallelised, e.g. set it to `"mpirun -n 4 lmp -in lmp.in"` to run every LAMMPS calculation on 4 MPI ranks.

## How to submit a melting point calculation to an HPC queue?
[executorlib](https://github.com/pyiron/executorlib) also provides cluster executors which submit each task as its own job to a queuing system instead of running it on the local machine. Replace `executorlib.SingleNodeExecutor` in the notebook with `executorlib.SlurmClusterExecutor` (for SLURM) or `executorlib.FluxClusterExecutor` (for the [flux](https://flux-framework.org) framework) to submit every LAMMPS run individually to the queue, including the requested number of cores per task via the `resource_dict` argument. See the [executorlib documentation](https://github.com/pyiron/executorlib) for the full list of available executors and their configuration options.

# Acknowledgments
If you use the melting point protocol in your scientific work, please consider citing:
```
  @article{melting,
    title = {A fully automated approach to calculate the melting temperature of elemental crystals},
    journal = {Computational Materials Science}
    volume = {187},
    pages = {110065},
    year = {2021},
    doi = {https://doi.org/10.1016/j.commatsci.2020.110065},
    url = {https://www.sciencedirect.com/science/article/pii/S0927025620305565},
    author = {Li-Fang Zhu and Jan Janssen and Shoji Ishibashi and Fritz Körmann and Blazej Grabowski and Jörg Neugebauer},
    keywords = {Interface method, Melting point, Arbitrary potential, pyiron},
  }

  @article{pyiron,
    title = {pyiron: An integrated development environment for computational materials science},
    journal = {Computational Materials Science},
    volume = {163},
    pages = {24 - 36},
    year = {2019},
    issn = {0927-0256},
    doi = {https://doi.org/10.1016/j.commatsci.2018.07.043},
    url = {http://www.sciencedirect.com/science/article/pii/S0927025618304786},
    author = {Jan Janssen and Sudarsan Surendralal and Yury Lysogorskiy and Mira Todorova and Tilmann Hickel and Ralf Drautz and Jörg Neugebauer},
    keywords = {Modelling workflow, Integrated development environment, Complex simulation protocols},
  }
  
  @article{pyscal,
    author = {Sarath Menon and Grisell Díaz Leines and Jutta Rogal},
    title = {{pyscal: A python module for structural analysis of atomic environments}},
    year = 2019,
    journal = {Journal of Open Source Software},
    doi = {10.21105/joss.01824},
    url = {https://doi.org/10.21105/joss.01824},
    volume = {4},
    number = {43},
    page = {1824}
  }

  @article{executorlib,
    title = {Executorlib -- Up-scaling Python workflows for hierarchical heterogenous high-performance computing},
    journal = {Journal of Open Source Software},
    volume = {10},
    number = {108},
    pages = {7782},
    year = {2025},
    issn = {2475-9066},
    doi = {10.21105/joss.07782},
    url = {https://joss.theoj.org/papers/10.21105/joss.07782},
    author = {Jan Janssen and Michael Gilbert Taylor and Ping Yang and Joerg Neugebauer and Danny Perez},
    keywords = {High-performance computing, Workflow management, Python},
  }
```

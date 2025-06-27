# A generalized method for refining and selecting random crystal structures using graph theory (QGBR)

## Dataset
Data supporting this research are available at [Zenodo] (https://doi.org/10.5281/zenodo.15688744).

## Requirements
Required Python packages include:  
- `ase`
- `numpy`
- `pymatgen`
- `spglib==2.0.2`
- `networkx`

## Run
Execute the following command to refine initial structures using QGBR. Make sure to substitute `input.yaml` with the actual path to input files:
- `python qgbr_refine.py --input_para input.yaml`

## Postprocessing
After QGBR, sereval files in `traj` format will be generated in the current directory, depending on the number of `num_workers` you have set. Execute the following command to merge these files into one file:
- `python post_process.py --num_workers 4`
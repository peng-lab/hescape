## Experiments

We use `hydra` for experiment management. The experiments are structured as follow:

-   In `configs` there are default and experiment specific configurations file.
-   Each experiment has a config file named after it, as well as a folder with the same name, for example:
    -   `./configs/spatialclip_pretrain.yaml` and `./spatialclip_pretrain` represent the same experiment.
        The first is the configuration file, the second is the folder where slurm logs and results are stored.

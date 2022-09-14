# COMP90089 Final Project Group 13

Group members
- Bushra Aldosari
- Joshua Hoyle
- Yvonne Tao
- Johnson Zhou

## Organisation

- `template`: template code to use for queries
- `mimic`: MIMIC exploration scripts
- `resource`: any resource material associated with the project

## Resources

MIMIC docs
- Github [MIT-LCP/mimic-iv-website][1] (more comprehensive)
- MIMIC-IV docs [website][2]

## Conda environments

`conda_env_gbq.yml`: To access and query MIMIC on Google BigQuery locally.  
`conda_env_tensor_m1.yml`: To enable tensorflow with GPU on M1 macs.

```bash
conda env create -f <file>.yml
```

## Visual Studio Code

Local access to MIMIC and BigQuery via the following extension:  
`https://marketplace.visualstudio.com/items?itemName=GoogleCloudTools.cloudcode`  


[1]: https://github.com/MIT-LCP/mimic-iv-website
[2]: https://mimic.mit.edu/docs/iv/
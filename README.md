# Phenotype discovery in patients exhibiting extended length of stay (LOS) in ICU settings

COMP90089 Machine Learning Applications in Health Final Project.  
Group #13, Semester 2, 2022.

Group members
- [Bushra Aldosari](https://github.com/baldosari)  
- [Joshua Hoyle](https://github.com/joshuahoyle)  
- [Yvonne Tao](https://github.com/Momotaro-Yvv)  
- [Johnson Zhou](https://github.com/johnsonjzhou)  

## Quick access supplementary information to the report  

Extracted datasets have been excluded from the repository as per PhysioNet's [data use agreement][3].  

They can be re-constructed by running the jupyter notebooks with credentialled access to MIMIC.  

Notebooks:  
- [Feature extraction SQL](./sql/initial_cohort_final_updated_los_bins.ipynb)
- [Data exploration](./data_exploration/exploratory_data_analysis_v2.ipynb)
- [Data processing](./data_exploration/data_processing_v2_full_cohort.ipynb)
- [Dimensionality Reduction PCA](./cluster/pca_v2_full_cohort.ipynb)
- [Clustering K-Means](./cluster/kmeans_v2_pca_full_cohort.ipynb)
- [Clustering DBSCAN](./cluster/dbscan_v2_pca_full_cohort.ipynb)
- [Feature importance: largest cluster](./feature_importance_analysis/fia_1stCluster_full_cohort.ipynb)
- [Feature importance: second largest cluster](./feature_importance_analysis/fia_2ndCluster_full_cohort.ipynb)

## Organisation

- `template`: template code to use for queries  
- `mimic`: MIMIC exploration scripts  
- `data`: data sources used in the project  
- `sql`: sql scripts used to query MIMIC  
- `data_exploration`: exploratory data analysis and data processing  
- `cluster`: scripts associated with dimensionality reduction and clustering  
- `feature_importance_analysis`: scripts associated with feature importance analysis
- `img`: extracted images and plots from notebooks  
- `src`: python helper functions

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

## Licensing

Usage of data within this repository must comply with PhysioNet's [data usage policy][3].

[1]: https://github.com/MIT-LCP/mimic-iv-website
[2]: https://mimic.mit.edu/docs/iv/
[3]: https://physionet.org/content/mimiciv/view-dua/2.0/
# Mechanistic Uncertainty Calibration

These files were deleted from the repository as they should be downloaded independently rather than redistributed:

* /data/KUQ/KUQ.ncan.dev.json
* /data/KUQ/KUQ.ncan.test.json
* /data/KUQ/KUQ.ncan.train.json
* /data/SelfAware/SelfAware.ncan.dev.json
* /data/SelfAware/SelfAware.ncan.test.json
* /data/SelfAware/SelfAware.ncan.train.json

## Intro: TODO



## Steps for reproducing results

1. Run qa-generate.py through qa-generate.sh after updating any hard-wired default paths (e.g. for the KUQ and SelfAware datasets and/or passing the correct --results_dir argument in qa-generate.sh. Ensure that the sbatch --array argument in qa-generate.sh matches in size the total number of triples (model,dataset,chunk) as specified in qa-generate.py variables model_names, dataset_names, and n_chunk. E.g. 1 model x 2 datasets x 10 chunks --> --array=0-19

```
sbatch qa-generate.sh
```

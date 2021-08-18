# Evolving Search Space for Neural Architecture Search


# Usage

Install all required dependencies in requirements.txt and replace all `..path/..to` in the code to the absolute path 
to corresponding resources. The LUT (Latency Lookup Table) and meta files of ImageNet we used for searching are 
placed in  `resources` folder.

Only slurm based distributed training is implemented.

#### Searching

All experiment files are located in `experiment/NSE` folder.

To search for NSENET-27 on ImageNet, run 

```bash
sh search_NSE27.sh <number-of-nodes> <gpu-partition> 
```

To search for NSENET on ImageNet, run 
```bash
sh search_NSE_second_space.sh <number-of-nodes> <gpu-partition> 
```

To search for NSENET-GPU on ImageNet, run 
```bash
sh search_NSE_GPU.sh <number-of-nodes> <gpu-partition> 
```

#### Searched models

Final models (NSENET-27, NSENET, NSENET-GPU) are placed in `utils/__init__.py`.

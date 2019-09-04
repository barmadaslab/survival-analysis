# Survival Analysis

## Purpose
This system will identify neurons within fluorescent microscopy images, track them, determine their fate, and output relevant group hazard ratios by invoking an R script that runs Cox Proportional Hazards. The code has been uploaded in support of [Matrin 3-dependent neurotoxicity is modified by nucleic acid binding and nucleocytoplasmic localization](https://elifesciences.org/articles/35977). Its intent is to document the computational methods used within that paper. This code has been actively used within the laboratory and has multiple details specific to internal use that may not be relevant for your use case. To facilitate its interpretation, example data and some documentation is provided. 

## System details
### MFile
Internal lab experiments are currently documented via what we've termed Mfiles. These files serve as a manifest of experimental details and aid in configuring and running our automated microscopy system. This same document is then used to run our code. As such, this code uses an Mfile. This may be unnecessary for other applications and you may want to remove dependence on Mfiles within the code for your own purposes. All Mfile parsing code is within the _mfileutils_ module. Please refer to */mfileutils/makeconfig.py* for specific parsing details.

### Image resolution
Multiple cameras are used within our lab and each has different resolution. These specific parameters, which will likely be different from your own, are set within the _imgutils/transforms_ module.

### File structure
A file structure unique to the lab's processes and overall workflow is used. You may want to modify this for your own purposes.

## Running survival analysis on example data
4 example stacks are within the <i>_example_data</i>. The essence of the survival analysis system is within _analysis.py_. Open this and execute it with Python. 

### What results to expect
An example of the console output is within <i>_example_console_out.png</i>. 

Within <i>_example_data_</i> you will already find *analysis* and *results*  directories, which have been uploaded for users to see example results without having to run the system. The *annotated*  stacks for the example data exceed the current upload limit, so they have been zipped before uploading. Both */analysis* and */results*  are created by the program, and all of its contents are output during execution.

#### Annotated stacks
These are output to <i>_example_data/analysis/annotated/</i>. These images are for humans to see and evaluate the tracking and survival analysis results. 

#### Results
Within <i>_example_data/results</i> will be a PDF file with an output hazard plot, a text file with numeric values for hazard ratios output by R, a survival data CSV, an *rois*  directory containing Python pickle files for each ROI found, and an *IJ_rois* directory that contains ROIs in ImageJ format. The ImageJ ROIs can be imported into ImageJ. The encoding was reverse engineered from its online source code and that module is available in *IJEncoding.py*. 

Results for <i>_example_data</i> are already within <i>_example_data/results/</i>.

## Version Numbers
Below are version numbers last used for ensuring functionality. 
1. Python 3.6.7, although Python3+ should work, with possible minimal modification.
2. Numpy 1.15.4
3. Pandas 0.20.1
4. Scipy 0.19.1
5. Skimage 0.13.0
6. Yaml 3.12
7. OpenCV 3.3.0
8. Multiprocess 0.70.5
9. [Rscript](https://www.rdocumentation.org/packages/utils/versions/3.6.1/topics/Rscript) 3.3.2 (not a Python module, but a means to access R via the command line)
10. [R package: survival](https://cran.r-project.org/web/packages/survival/index.html)

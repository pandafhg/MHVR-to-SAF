# MHVR-to-SAF

# Update. 2022-09-05
A batch processing program for MHVR_to_SAF has been created.
The following contents have been added to the folder.
- The file RunOnnxBatch.py is the batch processing.
- The folder MHVR contains all MHVR data (support *.csv, *.txt, *.dat) that needs to be processed. 
  Spacing of data columns supports spaces, double spaces, tabs and commas.
- The MHVR_SAF folder contains processed pSAF data and is named in the format [SAF_source_filename]. Some examples can be found in the MHVR_SAF_Sample folder.

Flow:

* Click on RunOnnxBatch.py and it will show the path where the pSAF is stored; 
* Enter the folder path (full path) with all MHVR data;.
* After the process is completed, the saved file path is shown.

# Introduction
This study proposed a novel methodology for directly estimating S-wave site amplification factors (SAF) from microtremor 
horizontal-to-vertical spectral ratio (MHVR) based on deep neural network (DNN) model. 

The DNN model was developed using peak frequency and the frequency-dependent relationship between MHVRs and SAFs.

The model can automatically produce site amplification factor of S-wave only from microtremor data.

# Documents
"model.onnx" : We provide the onnx file of already trained DNN model.

"RunOnnx.py" : A simple python-based code to run the model in this study with ONNX Runtime.

"sample.csv" : Input data for example.

"output.csv" : Output data for example.

"RunOnnx Tutorials.docx" : A detailed tutorial with English and Japanese versions.

# Related Article
For more detailed description, please refer to the authors' article (below).

If you need to use the model in your study, please cite the author's article (below).

Pan et al., Deep Neural Network-based Estimation of Site Amplification Factor from Microtremor H/V Spectral Ratio (https://doi.org/10.1785/0120210300).

# Contact
If you have any questions or suggestions, please send an email to the author.

Da Pan; pd19951229@gmail.com; d214903@hiroshima-u.ac.jp

import onnx
import numpy as np
import pandas as pd
import onnxruntime as rt
import os

# Iterate through paths of all files in the folder (fname). And save as a list (s_file).
def get_file(fname):
    s_file = []
    for path,dirs,files in os.walk(fname):
        for filename in files:
            s_file.append(os.path.join(path,filename))
    return s_file

# Retrieve DNN model file (model.onnx) from the current directory.
if os.path.exists("%s/model.onnx" % os.getcwd()):
    model_file = "%s/model.onnx" % os.getcwd()
# If the model file does not exist in the current directory, 
# please input the full path of the model.onnx.
else:
    model_file = input("input the file path of model.onnx (/*.onnx):")

# Set the save path. The default is the MHVR_SAF folder in the current directory.    
if os.path.exists("%s/MHVR_SAF" % os.getcwd()):
    y_folder = "%s/MHVR_SAF" % os.getcwd() 
    print("The result will be saved in this path: %s" % y_folder)
else:
    os.mkdir("%s/MHVR_SAF" % os.getcwd())
    y_folder = "%s/MHVR_SAF" % os.getcwd()
    print("The result will be saved in this path: %s" % y_folder)

# Input the path of folder containing MHVR data files.
mhvr_loc = input("input the path of the folder containing MHVR data files: ")


for test_file in get_file(mhvr_loc):
    X_test = pd.read_csv(test_file ,skiprows=1, sep = '\s+|,', header =None, engine='python') # encoding = "utf-8"
    X_test = X_test.values
    
    sess = rt.InferenceSession(model_file)
    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[0].name

    Y_result = pd.DataFrame()
    for row in X_test:
        pred_onx = sess.run([label_name], {input_name: row.reshape(1,7).astype(np.float32)})[0]
        Y_result = pd.concat([Y_result,pd.DataFrame(pred_onx)], ignore_index= True)

    Y_result.columns=["AMR(fi-2)", "AMR(fi-1)", "AMR(fi)", "AMR(fi+1)", "AMR(fi+2)"]
    
    X_temp = pd.read_csv(test_file, sep = '\s+|,', engine='python') # encoding = "utf-8"

    result = pd.concat([X_temp, Y_result], axis=1)
    result["pSAF"] = result["%s" % X_temp.columns[4]] * result["AMR(fi)"]
    result.round(6)
    
    # Save by source filename. 
    name_file = os.path.basename(test_file)
    
    if os.path.splitext(name_file)[1] == '.csv':
        result.to_csv("%s/SAF_%s" % (y_folder, name_file), float_format = '%.6f', sep = ',',index=None)
        
    else:     
        result.to_csv("%s/SAF_%s" % (y_folder, name_file), float_format = '%.6f', sep = '\t',index=None)
        
    print("Saved as %s/SAF_%s" % (y_folder, name_file))
input("   ")
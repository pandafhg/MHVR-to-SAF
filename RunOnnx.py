# Refrence:https://www.onnxruntime.ai/python/index.html
# We provide onnx files of already trained DNN models.
# A simple code to run the model in this study with ONNX Runtime.
# Make sure the csv file encoding is utf-8, otherwise it may report error

# input test.csv file: sample.csv
# input model.onnx file: model_fold_1.onnx
# The result is saved as: output.csv

import onnx
import numpy as np
import pandas as pd
import onnxruntime as rt


test_file = input("input test.csv file (*.csv):")
model_file = input("input model.onnx file (*.onnx):")
y_file = input("The result is saved as (*.csv):")

X_test = pd.read_csv(test_file ,skiprows=1, header =None, engine='python') # encoding = "utf-8"
X_test = X_test.values

sess = rt.InferenceSession(model_file)
input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name

Y_result = pd.DataFrame()
for row in X_test:
    pred_onx = sess.run([label_name], {input_name: row.reshape(1,7).astype(np.float32)})[0]
    Y_result = Y_result.append(pd.DataFrame(pred_onx), ignore_index = True)
    
Y_result.columns=["AMR(fi-2)", "AMR(fi-1)", "AMR(fi)", "AMR(fi+1)", "AMR(fi+2)"]
X_temp = pd.read_csv(test_file) # encoding = "utf-8"

result = pd.concat([X_temp, Y_result], axis=1)
result["pSAF"] = result["%s" % X_temp.columns[4]] * result["AMR(fi)"]
print(result)

result.to_csv(y_file, float_format = '%.6f', index=None)

print("Saved as %s" % y_file)
input("   ")

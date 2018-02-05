import tensorflow as tf


path = "checkpoints/model1/model"
reader = tf.train.NewCheckpointReader(path)

all_variables = reader.get_variable_to_shape_map()

for variables_name in all_variables:
    print(variables_name, all_variables[variables_name])  #变量名， 变量维度
print("value of variable v1 is ", reader.get_tensor('v1'))
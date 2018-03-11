# Usual imports
import numpy as np, tensorflow as tf

# First operand (the name attribute is optionnal)
op1 = tf.constant(40, name="Constant_A")

# Second operand
op2 = tf.constant(2, name="Constant_B")

# Operation
totalsum = tf.add(op1, op2)

# Not what we expect!
print("Result?", totalsum)

# We compile the graph
sess = tf.Session()

# Logs the graph and results
# This is not mandatory to get the result, but allows the use of TensorBoard
#writer = tf.train.SummaryWriter("/tmp/tflogs_1A", sess.graph_def, flush_secs=10)
summary_writer = tf.summary.FileWriter("./exp_A/tflogs", sess.graph)

# and get the result
result = sess.run(totalsum)

# Tada!
print("Actual result!", result)

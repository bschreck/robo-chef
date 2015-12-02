import tensorflow as tf
# Create a Variable, that will be initialized to the scalar value 0.
var = tf.Variable(0, name="counter")

# Create an Op to add one to `var`.

one = tf.constant(1)
new_value = tf.add(var, one)
update = tf.assign(var, new_value)

# Variables must be initialized by running an `init` Op after having

# launched the graph.  We first have to add the `init` Op to the graph.
init_op = tf.initialize_all_variables()

# Launch the graph and run the ops.
with tf.Session() as sess:
    # Run the 'init' op
    sess.run(init_op)
    # Print the initial value of 'var'
    print sess.run(var)
    # Run the op that updates 'var' and print 'var'.
    for _ in range(3):
        sess.run(update)
        print sess.run(var)

input1 = tf.constant(3.0)
input2 = tf.constant(2.0)
input3 = tf.constant(5.0)
intermed = tf.add(input2, input3)
mul = tf.mul(input1, intermed)

with tf.Session() as sess:
    result = sess.run([mul, intermed])
    print result

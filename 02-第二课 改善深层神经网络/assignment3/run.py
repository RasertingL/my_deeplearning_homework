import math
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
import h5py
import matplotlib.pyplot as plt
import pylab
import scipy
from PIL import Image
from scipy import ndimage
import pickle as pkl

from tf_utils import load_dataset, random_mini_batches, convert_to_one_hot, predict

#%matplotlib inline
np.random.seed(1)

#创建一个1*2的矩阵
matrix1 = tf.constant([[3,1]])
#创建一个2*1的矩阵
matrix2 = tf.constant([[1] , [5]])
#将他们相乘
produce = tf.matmul(matrix1 , matrix2)

sess = tf.Session()
print(sess.run(produce))
#会话关闭
sess.close()


#现在试试对于变量的操作
#我们先来做一个计数器
#先创建一个变量初始化为0
state = tf.Variable(0 , name="counter")
# 创建一个op节点，初始化为1，作用是让变量每次加一
one = tf.constant(1)
new_value = tf.add(state , one)
update = tf.assign(state , new_value) #注意，这里暂时不知道什么意思，官方文档的实例里暂时没看到
# 启动图后, 变量必须先经过`初始化` (init) op 初始化,
# 首先必须增加一个`初始化` op 到图中.
init_op = tf.initialize_all_variables()

# 代码中 assign() 操作是图所描绘的表达式的一部分, 正如 add() 操作一样. 所以在调用 run() 执行表达式
# 之前, 它并不会真正执行赋值操作.
# 通常会将一个统计模型中的参数表示为一组变量. 例如, 你可以将一个神经网络的权重作为某个变量存储在一个
# tensor 中. 在训练过程中, 通过重复运行训练图, 更新这个 tensor

#启动会话
with tf.Session() as sess :
    # 运行'init'op
    sess.run(init_op)
    # 打印'state'的初始值
    print(sess.run(state))
    print(sess.run(new_value))
    # 运行'op'，更新'state'，并打印'state'
    for _ in range(3):
        sess.run(update)
        print(sess.run(state))

#上方为自己看官方文档做的小练习，下面开始做作业。。。


#完成lost损失函数的运算
y_hat = tf.constant(36, name='y_hat')            # Define y_hat constant. Set to 36.是一个常量
y = tf.constant(39, name='y')                    # Define y. Set to 39

loss = tf.Variable((y - y_hat)**2, name='loss')  # Create a variable for the loss是一个变量

init = tf.global_variables_initializer()         # When init is run later (session.run(init)),
                                                 # the loss variable will be initialized and ready to be computed
with tf.Session() as session:                    # Create a session and print the output
    session.run(init)                            # Initializes the variables
    print(session.run(loss))                     # Prints the loss

# Writing and running programs in TensorFlow has the following steps:
#
# 1. Create Tensors (variables) that are not yet executed/evaluated.
# 2. Write operations between those Tensors.
# 3. Initialize your Tensors.
# 4. Create a Session.
# 5. Run the Session. This will run the operations you'd written above.
#
# Therefore, when we created a variable for the loss, we simply defined the loss as a function of other quantities, but did not evaluate its value. To evaluate it, we had to run `init=tf.global_variables_initializer()`. That initialized the loss variable, and in the last line we were finally able to evaluate the value of `loss` and print its value.

a = tf.constant(2)
b = tf.constant(10)
c = tf.multiply(a,b)
sess = tf.Session()
print(sess.run(c))
print("=======================")


#下面来看看tf.placeholder函数
# Change the value of x in the feed_dict
# A placeholder is an object whose value you can specify only later.
# To specify values for a placeholder, you can pass in values by using a "feed dictionary" (`feed_dict` variable).
# Below, we created a placeholder for x.
# This allows us to pass in a number later when we run the session.
x = tf.placeholder(tf.int64, name = 'x')
print(sess.run(2 * x, feed_dict = {x: 3})) #这里在sess.run里才对x进行赋值
sess.close()
print("========================")

# #对于下面构造的函数，你可能需要用到这些
# - tf.matmul(..., ...) to do a matrix multiplication
# - tf.add(..., ...) to do an addition
# - np.random.randn(...) to initialize randomly
# GRADED FUNCTION: linear_function

def linear_function():
    """
    Implements a linear function:
            Initializes W to be a random tensor of shape (4,3)
            Initializes X to be a random tensor of shape (3,1)
            Initializes b to be a random tensor of shape (4,1)
    Returns:
    result -- runs the session for Y = WX + b
    """

    np.random.seed(1)

    ### START CODE HERE ### (4 lines of code)
    X = tf.constant(np.random.randn(3 , 1) , name="X")
    W = tf.constant(np.random.randn(4 , 3) , name="W")
    b = tf.constant(np.random.randn(4 , 1) , name="b")
    Y = tf.add(tf.matmul(W , X) , b)
    ### END CODE HERE ###

    # Create the session using tf.Session() and run it with sess.run(...) on the variable you want to calculate

    ### START CODE HERE ###
    sess = tf.Session()
    result = sess.run(Y)
    ### END CODE HERE ###

    # close the session
    sess.close()

    return result

#下面来测试一下
print( "result = " + str(linear_function()))
print("=================================")


#下面这里又有点好奇，这是为了熟悉而进行的训练么？难道使用的时候在主函数里直接tf.sigmoid(x)不会跑。。？（貌似tf.sigmoid()也可以）
# GRADED FUNCTION: sigmoid
def sigmoid(z):
    """
    Computes the sigmoid of z

    Arguments:
    z -- input value, scalar or vector

    Returns:
    results -- the sigmoid of z
    """

    ### START CODE HERE ### ( approx. 4 lines of code)
    # Create a placeholder for x. Name it 'x'.
    x = tf.placeholder(tf.float32 , name="x")

    # compute sigmoid(x)
    sigmoid = tf.sigmoid(x)

    # Create a session, and run it. Please use the method 2 explained above.
    # You should use a feed_dict to pass z's value to x.
    with tf.Session() as sess:

        # Run session and call the output "result"
        result = sess.run(sigmoid, feed_dict={x : z})

    ### END CODE HERE ###

    return result

#输出看看
print ("sigmoid(0) = " + str(sigmoid(0)))
print ("sigmoid(12) = " + str(sigmoid(12)))
print("===============================")


#来试试用TensorFlow写cost函数
# GRADED FUNCTION: cost
def cost(logits, labels):
    """
    Computes the cost using the sigmoid cross entropy
    
    Arguments:
    logits -- vector containing z, output of the last linear unit (before the final sigmoid activation)
    labels -- vector of labels y (1 or 0)

    Note: What we've been calling "z" and "y" in this class are respectively called "logits" and "labels"
    in the TensorFlow documentation. So logits will feed into z, and labels into y.
    
    Returns:
    cost -- runs the session of the cost (formula (2))
    """

    ### START CODE HERE ###

    # Create the placeholders for "logits" (z) and "labels" (y) (approx. 2 lines)
    z = tf.placeholder(tf.float32 , name="z")
    y = tf.placeholder(tf.float32 , name="y")

    # Use the loss function (approx. 1 line)
    cost = tf.nn.sigmoid_cross_entropy_with_logits(logits = z , labels = y)

    # Create a session (approx. 1 line). See method 1 above.
    sess = tf.Session()

    # Run the session (approx. 1 line).
    cost = sess.run(cost , feed_dict={z : logits, y : labels})

    # Close the session (approx. 1 line). See method 1 above.
    sess.close()

    ### END CODE HERE ###

    return cost

#好，来跑一跑这个tf的神奇的cost函数（讲道理，我看到那条语句真的震惊。。。仿佛没见过世面）
logits = sigmoid(np.array([0.2,0.4,0.7,0.9]))
cost = cost(logits, np.array([0,0,1,1]))
print ("cost = " + str(cost))
print("===================")


# #下面谈谈独热编码（我看了这个英文文档，查了下这个单词，发现是第一次听说这个名词。。。是我太菜）
# 指的是在分类问题中，将存在数据类别的那一类用X表示，不存在的用Y表示，这里的X常常是1， Y常常是0。，举个例子：
# 比如我们有一个5类分类问题，我们有数据(xi,yi)(xi,yi)，其中类别yiyi有五种取值（因为是五类分类问题），所以如果yjyj为第一类那么其独热编码为：
# [1,0,0,0,0]，如果是第二类那么独热编码为：[0,1,0,0,0]，也就是说只对存在有该类别的数的位置上进行标记为1，其他皆为0。这个编码方式经常用于多分类问题，特别是损失函数为交叉熵函数的时候。
# GRADED FUNCTION: one_hot_matrix
def one_hot_matrix(labels, C):
    """
    Creates a matrix where the i-th row corresponds to the ith class number and the jth column
                     corresponds to the jth training example. So if example j had a label i. Then entry (i,j)
                     will be 1.

    Arguments:
    labels -- vector containing the labels
    C -- number of classes, the depth of the one hot dimension

    Returns:
    one_hot -- one hot matrix
    """

    ### START CODE HERE ###

    # Create a tf.constant equal to C (depth), name it 'C'. (approx. 1 line)
    C = tf.constant(C, name="C")

    # Use tf.one_hot, be careful with the axis (approx. 1 line)
    one_hot_matrix = tf.one_hot(labels , C , axis=0)

    # Create the session (approx. 1 line)
    sess = tf.Session()

    # Run the session (approx. 1 line)
    one_hot = sess.run(one_hot_matrix)

    # Close the session (approx. 1 line). See method 1 above.
    sess.close()

    ### END CODE HERE ###

    return one_hot

#现在来跑一下
labels = np.array([1,2,3,0,2,1])
one_hot = one_hot_matrix(labels, C = 4)
print ("one_hot = " + str(one_hot))
#注意这个对于Y是竖着的
print("=====================")


#现在创建矩阵，让里面的元素全是0或者1，用语句 tf.ones()
# GRADED FUNCTION: ones

def ones(shape):
    """
    Creates an array of ones of dimension shape

    Arguments:
    shape -- shape of the array you want to create

    Returns:
    ones -- array containing only ones
    """

    ### START CODE HERE ###

    # Create "ones" tensor using tf.ones(...). (approx. 1 line)
    ones = tf.ones(shape)

    # Create the session (approx. 1 line)
    sess = tf.Session()

    # Run the session to compute 'ones' (approx. 1 line)
    ones = sess.run(ones)

    # Close the session (approx. 1 line). See method 1 above.
    sess.close()

    ### END CODE HERE ###
    return ones

#让我们看看效果
print ("ones = " + str(ones([3])))
print("=======================")


#下面我们来永tf写一下机器学习的算法
# Loading the dataset
#先导入数据
X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()
#展示一张图片看看
# Example of a picture
index = 0
plt.imshow(X_train_orig[index])
pylab.show()
#看到一只手，然后它的y显示它是y=5的类
print ("y = " + str(np.squeeze(Y_train_orig[:, index])))
print("=========================")

#然后我们队数据进行处理，尤其是将其归一化(除以255)，并且将进行独热编码
# Flatten the training and test images
X_train_flatten = X_train_orig.reshape(X_train_orig.shape[0], -1).T
X_test_flatten = X_test_orig.reshape(X_test_orig.shape[0], -1).T
# Normalize image vectors
X_train = X_train_flatten/255.
X_test = X_test_flatten/255.
# Convert training and test labels to one hot matrices
Y_train = convert_to_one_hot(Y_train_orig, 6)
Y_test = convert_to_one_hot(Y_test_orig, 6)

print ("number of training examples = " + str(X_train.shape[1]))
print ("number of test examples = " + str(X_test.shape[1]))
print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(Y_train.shape))
print ("X_test shape: " + str(X_test.shape))
print ("Y_test shape: " + str(Y_test.shape))
#通过Y的列，我们可以看出，有6个类
print("=============================")


# 现在开始搭建一个神经网络，当然，是用tf，并且最后使用softmax函数
# GRADED FUNCTION: create_placeholders
def create_placeholders(n_x, n_y):
    """
    Creates the placeholders for the tensorflow session.

    Arguments:
    n_x -- scalar, size of an image vector (num_px * num_px = 64 * 64 * 3 = 12288)
    n_y -- scalar, number of classes (from 0 to 5, so -> 6)

    Returns:
    X -- placeholder for the data input, of shape [n_x, None] and dtype "float"
    Y -- placeholder for the input labels, of shape [n_y, None] and dtype "float"

    Tips:
    - You will use None because it let's us be flexible on the number of examples you will for the placeholders.
      In fact, the number of examples during test/train is different.
    """

    ### START CODE HERE ### (approx. 2 lines)
    X = tf.placeholder(tf.float32, shape = (n_x,None))
    Y = tf.placeholder(tf.float32, shape = (n_y,None))
    ### END CODE HERE ###

    return X, Y

#，我们已经建立X，Y，现在看看
X, Y = create_placeholders(12288, 6)
print ("X = " + str(X))
print ("Y = " + str(Y))
print("======================")


#下面我们开始初始化参数吧（注意看初始化的那个函数，暂时还不太清楚这样和普通的随机初始化有什么不同）
# GRADED FUNCTION: initialize_parameters
def initialize_parameters():
    """
    Initializes parameters to build a neural network with tensorflow. The shapes are:
                        W1 : [25, 12288]
                        b1 : [25, 1]
                        W2 : [12, 25]
                        b2 : [12, 1]
                        W3 : [6, 12]
                        b3 : [6, 1]

    Returns:
    parameters -- a dictionary of tensors containing W1, b1, W2, b2, W3, b3
    """

    tf.set_random_seed(1)  # so that your "random" numbers match ours

    ### START CODE HERE ### (approx. 6 lines of code)
    W1 = tf.get_variable("W1", [25 , 12288] , initializer = tf.contrib.layers.xavier_initializer())
    b1 = tf.get_variable("b1" , [25 , 1] , initializer = tf.zeros_initializer())
    W2 = tf.get_variable("W2" , [12 , 25] , initializer = tf.contrib.layers.xavier_initializer())
    b2 = tf.get_variable("b2", [12,1], initializer = tf.zeros_initializer())
    W3 = tf.get_variable("W3", [6,12], initializer = tf.contrib.layers.xavier_initializer())
    b3 = tf.get_variable("b3", [6,1], initializer = tf.zeros_initializer())
    ### END CODE HERE ###

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3}

    return parameters

# 下面输出看看
tf.reset_default_graph()
with tf.Session() as sess:
    parameters = initialize_parameters()
    print("W1 = " + str(parameters["W1"]))
    print("b1 = " + str(parameters["b1"]))
    print("W2 = " + str(parameters["W2"]))
    print("b2 = " + str(parameters["b2"]))
    # 我们可以看到，这些随机初始化好的参数并没有被赋值
    print("=======================")


# 老规矩，现在来看前向传播啦，注意中间用relu函数，最后用softmax函数
# GRADED FUNCTION: forward_propagation
def forward_propagation(X, parameters):
    """
    Implements the forward propagation for the model: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SOFTMAX

    Arguments:
    X -- input dataset placeholder, of shape (input size, number of examples)
    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3"
                  the shapes are given in initialize_parameters

    Returns:
    Z3 -- the output of the last LINEAR unit
    """

    # Retrieve the parameters from the dictionary "parameters"
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']

    ### START CODE HERE ### (approx. 5 lines)              # Numpy Equivalents:
    Z1 = tf.add(tf.matmul(W1 , X) , b1)  # Z1 = np.dot(W1, X) + b1
    A1 = tf.nn.relu(Z1)  # A1 = relu(Z1)
    Z2 = tf.add(tf.matmul(W2 , A1) , b2)  # Z2 = np.dot(W2, a1) + b2
    A2 = tf.nn.relu(Z2)  # A2 = relu(Z2)
    Z3 = tf.add(tf.matmul(W3 , A2) , b3)  # Z3 = np.dot(W3,Z2) + b3
    ### END CODE HERE ###

    return Z3

# 前向传播完成，现在来试试
tf.reset_default_graph()
with tf.Session() as sess:
    X, Y = create_placeholders(12288, 6)
    parameters = initialize_parameters()
    Z3 = forward_propagation(X, parameters)
    print("Z3 = " + str(Z3))

    # 不知道大家发现没，这里竟然没有要我们缓存各层的Z和A，那我们待会反向传播咋办？或许这就是tf的神奇之处。。。


# 下面来算cost函数值
# GRADED FUNCTION: compute_cost
def compute_cost(Z3, Y):
    """
    Computes the cost

    Arguments:
    Z3 -- output of forward propagation (output of the last LINEAR unit), of shape (6, number of examples)
    Y -- "true" labels vector placeholder, same shape as Z3

    Returns:
    cost - Tensor of the cost function
    """

    # to fit the tensorflow requirement for tf.nn.softmax_cross_entropy_with_logits(...,...)
    logits = tf.transpose(Z3)
    labels = tf.transpose(Y)

    ### START CODE HERE ### (1 line of code)
    #注意这里是用的softmax函数
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits , labels = labels))
    ### END CODE HERE ###

    return cost

# 还是稳啊，cost直接一行
tf.reset_default_graph()

with tf.Session() as sess:
    X, Y = create_placeholders(12288, 6)
    parameters = initialize_parameters()
    Z3 = forward_propagation(X, parameters)
    cost = compute_cost(Z3, Y)
    print("cost = " + str(cost))
    print("==================")


# 这里为了将parameters键值对储存下来，自己写一个将其储存到.h5文件你的函数
def svaeparameters (parameters) :
    #创建(或打开？？？)文件
    file = h5py.File('saveparameters.h5' , 'w')
    #写入
    file.create_dataset('W1' , data=parameters["W1"])
    file.create_dataset('b1' , data=parameters["b1"])
    file.create_dataset('W2', data=parameters["W2"])
    file.create_dataset('b2', data=parameters["b2"])
    file.close()




#好了，现在还要反向传播，但是，你懂的，体现框架的好处的时候来了，
# 这个反向传播只有一行，，，所以，就不用另外弄一个函数了，直接放model里咯。
# 另外，说明一下在其中的反向传播过程：
# After you compute the cost function. You will create an "`optimizer`" object. You have to call this object along with the cost when running the tf.session. When called, it will perform an optimization on the given cost with the chosen method and learning rate.
# 意思是这个optimizer里面存的是把cost最小化并且乘以了学习率
# For instance, for gradient descent the optimizer would be:
# ```python
# optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate).minimize(cost)
# ```
#
# To make the optimization you would do:
# ```python
#   _ , c = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})
# ```
# 对于这个'_' 是一个习惯用法：把之后不需要用到的结果变量用'_'储存，表示这个值之后是不需要用到的


def model(X_train, Y_train, X_test, Y_test, learning_rate=0.0001,
          num_epochs=1500, minibatch_size=32, print_cost=True):
    """
    Implements a three-layer tensorflow neural network: LINEAR->RELU->LINEAR->RELU->LINEAR->SOFTMAX.

    Arguments:
    X_train -- training set, of shape (input size = 12288, number of training examples = 1080)
    Y_train -- test set, of shape (output size = 6, number of training examples = 1080)
    X_test -- training set, of shape (input size = 12288, number of training examples = 120)
    Y_test -- test set, of shape (output size = 6, number of test examples = 120)
    learning_rate -- learning rate of the optimization
    num_epochs -- number of epochs of the optimization loop
    minibatch_size -- size of a minibatch
    print_cost -- True to print the cost every 100 epochs

    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """

    ops.reset_default_graph()  # to be able to rerun the model without overwriting tf variables
    tf.set_random_seed(1)  # to keep consistent results
    seed = 3  # to keep consistent results
    (n_x, m) = X_train.shape  # (n_x: input size, m : number of examples in the train set)
    n_y = Y_train.shape[0]  # n_y : output size
    costs = []  # To keep track of the cost

    # Create Placeholders of shape (n_x, n_y)
    ### START CODE HERE ### (1 line)
    X, Y = create_placeholders(n_x , n_y)
    ### END CODE HERE ###

    # Initialize parameters
    ### START CODE HERE ### (1 line)
    #这里由于上面已经定了各层神经网络的规模，所以不需要参数(下次自己用这里得改)
    parameters = initialize_parameters()
    ### END CODE HERE ###

    # Forward propagation: Build the forward propagation in the tensorflow graph
    ### START CODE HERE ### (1 line)
    Z3 = forward_propagation(X , parameters)
    ### END CODE HERE ###

    # Cost function: Add cost function to tensorflow graph
    ### START CODE HERE ### (1 line)
    cost = compute_cost(Z3 , Y)
    ### END CODE HERE ###

    # Backpropagation: Define the tensorflow optimizer. Use an AdamOptimizer.
    ### START CODE HERE ### (1 line)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    ### END CODE HERE ###

    # Initialize all the variables
    init = tf.global_variables_initializer()

    # Start the session to compute the tensorflow graph
    with tf.Session() as sess:

        # Run the initialization
        sess.run(init)

        # Do the training loop
        # 这是训练循环次数
        for epoch in range(num_epochs):

            epoch_cost = 0.  # Defines a cost related to an epoch
            #看看用mini-batch得分多少组
            num_minibatches = int(m / minibatch_size)  # number of minibatches of size minibatch_size in the train set
            seed = seed + 1
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)

            # 这里是使用mini-batch，将训练样本分组
            for minibatch in minibatches:
                # Select a minibatch
                (minibatch_X, minibatch_Y) = minibatch

                # IMPORTANT: The line that runs the graph on a minibatch.
                # Run the session to execute the "optimizer" and the "cost", the feedict should contain a minibatch for (X,Y).
                ### START CODE HERE ### (1 line)
                _, minibatch_cost = sess.run([optimizer , cost] , feed_dict={X : minibatch_X , Y : minibatch_Y})
                ### END CODE HERE ###

                epoch_cost += minibatch_cost / num_minibatches #有个问题，最后那个组的大小一般会小于其他的组，那这个比例岂不是会更大

            # Print the cost every epoch
            if print_cost == True and epoch % 100 == 0:
                print("Cost after epoch %i: %f" % (epoch, epoch_cost))
            if print_cost == True and epoch % 5 == 0:
                costs.append(epoch_cost)

        # plot the cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

        # lets save the parameters in a variable
        parameters = sess.run(parameters)
        print("Parameters have been trained!")

        # Calculate the correct predictions
        correct_prediction = tf.equal(tf.argmax(Z3), tf.argmax(Y))

        # Calculate accuracy on the test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        print("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train}))
        print("Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test}))

        return parameters

print("现在开始训练参数(要一会噢，五六分钟吧)")
parameters = model(X_train, Y_train, X_test, Y_test)

#保存好训练好的参数
file = open('saveparameters.txt', 'w')
file.write(str(parameters))
file.close()

#
# #读取
# file = open('saveparameters.txt','r')
# saveparameters = file.read()
# parameters = eval(saveparameters)
# f.close()


#现在开始可以用自己的图来测试效果啦，用手比一个数字，然后就可以识别了：
## START CODE HERE ## (PUT YOUR IMAGE NAME)
my_image = "thumbs_up.jpg"
## END CODE HERE ##

# We preprocess your image to fit your algorithm.
fname = "images/" + my_image
image = np.array(ndimage.imread(fname, flatten=False))
my_image = scipy.misc.imresize(image, size=(64,64)).reshape((1, 64*64*3)).T
my_image_prediction = predict(my_image, parameters)

plt.imshow(image)
print("Your algorithm predicts: y = " + str(np.squeeze(my_image_prediction)))

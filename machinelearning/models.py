import numpy as np

import backend
import nn

class Model(object):
    """Base model class for the different applications"""
    def __init__(self):
        self.get_data_and_monitor = None
        self.learning_rate = 0.0

    def run(self, x, y=None):
        raise NotImplementedError("Model.run must be overriden by subclasses")

    def train(self):
        """
        Train the model.

        `get_data_and_monitor` will yield data points one at a time. In between
        yielding data points, it will also monitor performance, draw graphics,
        and assist with automated grading. The model (self) is passed as an
        argument to `get_data_and_monitor`, which allows the monitoring code to
        evaluate the model on examples from the validation set.
        """
        for x, y in self.get_data_and_monitor(self):
            graph = self.run(x, y)
            graph.backprop()
            graph.step(self.learning_rate)

class RegressionModel(Model):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        Model.__init__(self)
        self.get_data_and_monitor = backend.get_data_and_monitor_regression

        # Remember to set self.learning_rate!
        # You may use any learning rate that works well for your architecture
        "*** YOUR CODE HERE ***"
        self.learning_rate = 0.05
        self.m1 = nn.Variable(1,100)
        self.b1 = nn.Variable(100)
        self.m2 = nn.Variable(100,1)
        self.b2 = nn.Variable(1)


    def run(self, x, y=None):
        """
        Runs the model for a batch of examples.

        The correct outputs `y` are known during training, but not at test time.
        If correct outputs `y` are provided, this method must construct and
        return a nn.Graph for computing the training loss. If `y` is None, this
        method must instead return predicted y-values.

        Inputs:
            x: a (batch_size x 1) numpy array
            y: a (batch_size x 1) numpy array, or None
        Output:
            (if y is not None) A nn.Graph instance, where the last added node is
                the loss
            (if y is None) A (batch_size x 1) numpy array of predicted y-values

        Note: DO NOT call backprop() or step() inside this method!
        """
        "*** YOUR CODE HERE ***"
        # print(x.shape)
        # print(y.shape)

        graph = nn.Graph([self.m1, self.b1, self.m2, self.b2])
        input_x = nn.Input(graph, x)
        xm1 = nn.MatrixMultiply(graph, input_x, self.m1)
        xm1_plus_b1 = nn.MatrixVectorAdd(graph, xm1, self.b1)
        relu1 = nn.ReLU(graph, xm1_plus_b1)
        xm2 = nn.MatrixMultiply(graph, relu1, self.m2)
        xm2_plus_b2 = nn.MatrixVectorAdd(graph, xm2, self.b2)

        if y is not None:
            # At training time, the correct output `y` is known.
            # Here, you should construct a loss node, and return the nn.Graph
            # that the node belongs to. The loss node must be the last node
            # added to the graph.
            "*** YOUR CODE HERE ***"
            input_y = nn.Input(graph, y)
            loss2 = nn.SquareLoss(graph, xm2_plus_b2, input_y)
            graph.add(loss2)
            return graph
        else:
            # At test time, the correct output is unknown.
            # You should instead return your model's prediction as a numpy array
            "*** YOUR CODE HERE ***"
            return graph.get_output(xm2_plus_b2)
            #return graph



class OddRegressionModel(Model):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers.

    Unlike RegressionModel, the OddRegressionModel must be structurally
    constrained to represent an odd function, i.e. it must always satisfy the
    property f(x) = -f(-x) at all points during training.
    """
    def __init__(self):
        Model.__init__(self)
        self.get_data_and_monitor = backend.get_data_and_monitor_regression

        # Remember to set self.learning_rate!
        # You may use any learning rate that works well for your architecture
        "*** YOUR CODE HERE ***"
        self.learning_rate = 0.04
        self.m1 = nn.Variable(1,200)
        self.b1 = nn.Variable(200)
        self.m2 = nn.Variable(200,1)
        self.b2 = nn.Variable(1)


    def run(self, x, y=None):
        """
        Runs the model for a batch of examples.

        The correct outputs `y` are known during training, but not at test time.
        If correct outputs `y` are provided, this method must construct and
        return a nn.Graph for computing the training loss. If `y` is None, this
        method must instead return predicted y-values.

        Inputs:
            x: a (batch_size x 1) numpy array
            y: a (batch_size x 1) numpy array, or None
        Output:
            (if y is not None) A nn.Graph instance, where the last added node is
                the loss
            (if y is None) A (batch_size x 1) numpy array of predicted y-values

        Note: DO NOT call backprop() or step() inside this method!
        """
        "*** YOUR CODE HERE ***"
        negator = np.array([-1.0])
        graph = nn.Graph([self.m1, self.b1, self.m2, self.b2])


        input_x = nn.Input(graph, x)
        input_x2 = nn.Input(graph, x * (-1))

        xm1 = nn.MatrixMultiply(graph, input_x, self.m1)
        x2m1 = nn.MatrixMultiply(graph, input_x2, self.m1)

        xm1_plus_b1 = nn.MatrixVectorAdd(graph, xm1, self.b1)
        x2m1_plus_b1 = nn.MatrixVectorAdd(graph, x2m1, self.b1)

        relu1 = nn.ReLU(graph, xm1_plus_b1)
        relu2 = nn.ReLU(graph, x2m1_plus_b1)

        xm2 = nn.MatrixMultiply(graph, relu1, self.m2)
        x2m2 = nn.MatrixMultiply(graph, relu2, self.m2)

        xm2_plus_b2 = nn.MatrixVectorAdd(graph, xm2, self.b2)
        x2m2_plus_b2 = nn.MatrixVectorAdd(graph, x2m2, self.b2)

        negatorNode = nn.Input(graph, -1 * np.ones((1,1)))
        ne_x2m2_plus_b2 = nn.MatrixMultiply(graph, x2m2_plus_b2, negatorNode)

        tempR = nn.Add(graph, xm2_plus_b2, ne_x2m2_plus_b2)
        if y is not None:
            # At training time, the correct output `y` is known.
            # Here, you should construct a loss node, and return the nn.Graph
            # that the node belongs to. The loss node must be the last node
            # added to the graph.
            "*** YOUR CODE HERE ***"

            input_y = nn.Input(graph, y)
            loss = nn.SquareLoss(graph, tempR, input_y)

            graph.add(loss)
            return graph
        else:
            # At test time, the correct output is unknown.
            # You should instead return your model's prediction as a numpy array
            "*** YOUR CODE HERE ***"
            return graph.get_output(tempR)

class DigitClassificationModel(Model):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        Model.__init__(self)
        self.get_data_and_monitor = backend.get_data_and_monitor_digit_classification

        # Remember to set self.learning_rate!
        # You may use any learning rate that works well for your architecture
        "*** YOUR CODE HERE ***"
        self.learning_rate = 0.5
        self.m1 = nn.Variable(784, 300)
        self.b1 = nn.Variable(300)
        self.m2 = nn.Variable(300, 10)
        self.b2 = nn.Variable(10)


    def run(self, x, y=None):
        """
        Runs the model for a batch of examples.

        The correct labels are known during training, but not at test time.
        When correct labels are available, `y` is a (batch_size x 10) numpy
        array. Each row in the array is a one-hot vector encoding the correct
        class.

        Your model should predict a (batch_size x 10) numpy array of scores,
        where higher scores correspond to greater probability of the image
        belonging to a particular class. You should use `nn.SoftmaxLoss` as your
        training loss.

        Inputs:
            x: a (batch_size x 784) numpy array
            y: a (batch_size x 10) numpy array, or None
        Output:
            (if y is not None) A nn.Graph instance, where the last added node is
                the loss
            (if y is None) A (batch_size x 10) numpy array of scores (aka logits)
        """
        "*** YOUR CODE HERE ***"
        graph = nn.Graph([self.m1, self.b1, self.m2, self.b2])
        input_x = nn.Input(graph, x)
        xm1 = nn.MatrixMultiply(graph, input_x, self.m1)
        xm1_plus_b1 = nn.MatrixVectorAdd(graph, xm1, self.b1)
        relu1 = nn.ReLU(graph, xm1_plus_b1)
        xm2 = nn.MatrixMultiply(graph, relu1, self.m2)
        xm2_plus_b2 = nn.MatrixVectorAdd(graph, xm2, self.b2)

        if y is not None:
            "*** YOUR CODE HERE ***"
            input_y = nn.Input(graph, y)
            loss2 = nn.SoftmaxLoss(graph, xm2_plus_b2, input_y)
            graph.add(loss2)
            return graph
        else:
            "*** YOUR CODE HERE ***"
            return graph.get_output(xm2_plus_b2)


class DeepQModel(Model):
    """
    A model that uses a Deep Q-value Network (DQN) to approximate Q(s,a) as part
    of reinforcement learning.

    (We recommend that you implement the RegressionModel before working on this
    part of the project.)
    """
    def __init__(self):
        Model.__init__(self)
        self.get_data_and_monitor = backend.get_data_and_monitor_rl

        self.num_actions = 2
        self.state_size = 4

        # Remember to set self.learning_rate!
        # You may use any learning rate that works well for your architecture
        "*** YOUR CODE HERE ***"
        self.learning_rate = 0.005
        self.m1 = nn.Variable(4, 100)
        self.b1 = nn.Variable(100)
        self.m2 = nn.Variable(100, 100)
        self.b2 = nn.Variable(100)
        self.m3 = nn.Variable(100, 2)
        self.b3 = nn.Variable(2)

    def run(self, states, Q_target=None):
        """

        """
        "*** YOUR CODE HERE ***"
        graph = nn.Graph([self.m1, self.b1, self.m2, self.b2, self.m3, self.b3])
        input_x = nn.Input(graph, states)

        xm1 = nn.MatrixMultiply(graph, input_x, self.m1)
        xm1_plus_b1 = nn.MatrixVectorAdd(graph, xm1, self.b1)
        relu1 = nn.ReLU(graph, xm1_plus_b1)

        xm2 = nn.MatrixMultiply(graph, relu1, self.m2)
        xm2_plus_b2 = nn.MatrixVectorAdd(graph, xm2, self.b2)
        relu2 = nn.ReLU(graph, xm2_plus_b2)

        xm3 = nn.MatrixMultiply(graph, relu2, self.m3)
        m3x_plus_b3 = nn.MatrixVectorAdd(graph, xm3, self.b3)

        if Q_target is not None:
            "*** YOUR CODE HERE ***"
            input_y = nn.Input(graph, Q_target)
            loss = nn.SquareLoss(graph, m3x_plus_b3, input_y)
            graph.add(loss)
            return graph
        else:
            "*** YOUR CODE HERE ***"
            result = graph.get_output(m3x_plus_b3)
            return result

    def get_action(self, state, eps):
        """
        Select an action for a single state using epsilon-greedy.

        Inputs:
            state: a (1 x 4) numpy array
            eps: a float, epsilon to use in epsilon greedy
        Output:
            the index of the action to take (either 0 or 1, for 2 actions)
        """
        if np.random.rand() < eps:
            return np.random.choice(self.num_actions)
        else:
            scores = self.run(state)
            return int(np.argmax(scores))


class LanguageIDModel(Model):
    """
    A model for language identification at a single-word granularity.

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        Model.__init__(self)
        self.get_data_and_monitor = backend.get_data_and_monitor_lang_id

        # Our dataset contains words from five different languages, and the
        # combined alphabets of the five languages contain a total of 47 unique
        # characters.
        # You can refer to self.num_chars or len(self.languages) in your code
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]

        # Remember to set self.learning_rate!
        # You may use any learning rate that works well for your architecture
        "*** YOUR CODE HERE ***"
        self.learning_rate = 0.0237

        size = 430
        self.m1 = nn.Variable(self.num_chars, size)
        self.b1 = nn.Variable(size)

        self.m2 = nn.Variable(size, size)
        self.b2 = nn.Variable(size)

        self.m3 = nn.Variable(size, 5)
        self.b3 = nn.Variable(5)

        self.h = nn.Variable(self.num_chars)
        self.mh = nn.Variable(self.num_chars, self.num_chars)
        self.bh = nn.Variable(self.num_chars)




    def run(self, xs, y=None):
        """
        Runs the model for a batch of examples.

        Although words have different lengths, our data processing guarantees
        that within a single batch, all words will be of the same length (L).

        Here `xs` will be a list of length L. Each element of `xs` will be a
        (batch_size x self.num_chars) numpy array, where every row in the array
        is a one-hot vector encoding of a character. For example, if we have a
        batch of 8 three-letter words where the last word is "cat", we will have
        xs[1][7,0] == 1. Here the index 0 reflects the fact that the letter "a"
        is the inital (0th) letter of our combined alphabet for this task.

        The correct labels are known during training, but not at test time.
        When correct labels are available, `y` is a (batch_size x 5) numpy
        array. Each row in the array is a one-hot vector encoding the correct
        class.

        Your model should use a Recurrent Neural Network to summarize the list
        `xs` into a single node that represents a (batch_size x hidden_size)
        array, for your choice of hidden_size. It should then calculate a
        (batch_size x 5) numpy array of scores, where higher scores correspond
        to greater probability of the word originating from a particular
        language. You should use `nn.SoftmaxLoss` as your training loss.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a (batch_size x self.num_chars) numpy array
            y: a (batch_size x 5) numpy array, or None
        Output:
            (if y is not None) A nn.Graph instance, where the last added node is
                the loss
            (if y is None) A (batch_size x 5) numpy array of scores (aka logits)

        Hint: you may use the batch_size variable in your code
        """
        batch_size = xs[0].shape[0]

        "*** YOUR CODE HERE ***"
        graph = nn.Graph([self.m1, self.b1, self.m2, self.b2, self.m3, self.b3, self.h, self.mh, self.bh])
        zero = nn.Input(graph, np.zeros((batch_size, self.num_chars)))
        h = nn.MatrixVectorAdd(graph, zero, self.h)

        for i in range(len(xs)):
            current_char = nn.Input(graph, xs[i])
            add_from_previous = nn.MatrixVectorAdd(graph, h, current_char)
            hmult = nn.MatrixMultiply(graph, add_from_previous, self.mh)
            hadd = nn.MatrixVectorAdd(graph, hmult, self.bh)
            hrelu = nn.ReLU(graph, hadd)
            h = hrelu

        m1h = nn.MatrixMultiply(graph, h, self.m1)
        m1h_add_b1 = nn.MatrixVectorAdd(graph, m1h, self.b1)
        relu1 = nn.ReLU(graph, m1h_add_b1)

        relu1_m2 = nn.MatrixMultiply(graph, relu1, self.m2)
        relu1_m2_add_b2 = nn.MatrixVectorAdd(graph, relu1_m2, self.b2)

        relu2 = nn.ReLU(graph, relu1_m2_add_b2)

        relu2_m3 = nn.MatrixMultiply(graph, relu2, self.m3)
        relu2_m3_add_b3 = nn.MatrixVectorAdd(graph, relu2_m3, self.b3)



        if y is not None:
            "*** YOUR CODE HERE ***"
            input_y = nn.Input(graph, y)
            loss = nn.SoftmaxLoss(graph, relu2_m3_add_b3, input_y)
            graph.add(loss)
            return graph

        else:
            "*** YOUR CODE HERE ***"
            return graph.get_output(relu2_m3_add_b3)

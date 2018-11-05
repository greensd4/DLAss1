import numpy as np
import utils as ut
import loglinear as ll

STUDENT={'name': 'Daniel Greenspan_Eilon Bashari',
         'ID': '308243948_308576933'}


def classifier_output(x, params):
    index = 0
    p = ut.params_to_couples(params)
    vec = x
    for (param1,param2) in p:
        # last params just (Wx + b)
        if index + 1 == len(p):
            vec = (np.dot(vec,param1)) + param2
            break
        # every layer params calculated as tanh(Wx + b)
        vec = np.tanh(np.dot(vec,param1) + param2)
        index += 1
    # all goes into softmax(x) for predictions vec
    probs = ll.softmax(vec)
    return probs


def predict(x, params):
    # return the prediction with the highest score.
    return np.argmax(classifier_output(x, params))


def calculate_vectors(x, params):
    # make the layers params
    tanh = []
    dots = []
    h = x
    tanh.append(h)
    for i in range(0, len(params), 2):
        z = np.dot(h,params[i]) + params[i + 1]
        dots.append(z)
        h = np.tanh(z)
        tanh.append(np.copy(h))
    # getting Wn and bn out of lists
    tanh.pop()
    dots.pop()
    return tanh, dots


def loss_and_gradients(x, y, params):
    """
    params: a list as created by create_classifier(...)

    returns:
        loss,[gW1, gb1, gW2, gb2, ...]

    loss: scalar
    gW1: matrix, gradients of W1
    gb1: vector, gradients of b1
    gW2: matrix, gradients of W2
    gb2: vector, gradients of b2
    ...

    (of course, if we request a linear classifier (ie, params is of length 2),
    you should not have gW2 and gb2.)
    """
    gradients = []
    tanhs, dots = calculate_vectors(x, params)
    probs = classifier_output(x, params)  # probabilities vec
    y_one_hot = np.zeros(len(probs))  # create one-hot vector
    y_one_hot[y] = 1

    # gradients values
    p = list(reversed(ut.params_to_couples(params)))

    d_l_z = -(y_one_hot-probs)

    h = tanhs.pop() #h n-1

    gW = np.outer(h,d_l_z)
    gb = np.copy(d_l_z)

    gradients.append(gW)
    gradients.append(gb)

    for i,(W, b) in enumerate(list(p)):

        if(len(dots) != 0):

            z = dots.pop()
            w_n_plus = W
            if(len(tanhs) != 0):
                d_z_W = tanhs.pop()
                d_h_z = 1 - np.square(np.tanh(z))
                d_z_h = w_n_plus
                d_l_z = np.dot(d_l_z, np.transpose(d_z_h)) * d_h_z

                gW = np.outer(d_z_W, d_l_z)
                gb = np.copy(d_l_z)

                gradients.append(gW)
                gradients.append(gb)


    gradients_in_asc_order = []
    for (W,b) in list(reversed(list(ut.params_to_couples(gradients)))):
        gradients_in_asc_order.append(W)
        gradients_in_asc_order.append(b)

    loss = -np.log(probs[y])
    return loss,gradients_in_asc_order


def create_classifier(dims):
    """
    returns the parameters for a multi-layer perceptron with an arbitrary number
    of hidden layers.
    dims is a list of length at least 2, where the first item is the input
    dimension, the last item is the output dimension, and the ones in between
    are the hidden layers.
    For example, for:
        dims = [300, 20, 30, 40, 5]
    We will have input of 300 dimension, a hidden layer of 20 dimension, passed
    to a layer of 30 dimensions, passed to learn of 40 dimensions, and finally
    an output of 5 dimensions.
    
    Assume a tanh activation function between all the layers.

    return:
    a flat list of parameters where the first two elements are the W and b from input
    to first layer, then the second two are the matrix and vector from first to
    second layer, and so on.
    """
    params = []
    root_six = np.sqrt(6)

    for in_dim,out_dim in ut.dim_to_couples(dims):
        eps = root_six / (np.sqrt(in_dim + out_dim))
        params.append(np.random.uniform(-eps, eps, [in_dim, out_dim]))
        eps = root_six / (np.sqrt(out_dim))
        params.append(np.random.uniform(-eps, eps, out_dim))
    return params


if __name__ == '__main__':

    from grad_check import gradient_check

    W, b, U, b_tag,V, b_t = create_classifier([3, 3, 4, 4])


    def _loss_and_W_grad(W):
        global b
        global U
        global b_tag
        global V
        global b_t
        loss, grads = loss_and_gradients([1, 2, 3], 0, [W, b, U, b_tag,V,b_t])
        return loss, grads[0]


    def _loss_and_b_grad(b):
        global W
        global U
        global b_tag
        global V
        global b_t
        loss, grads = loss_and_gradients([1, 2, 3], 0, [W, b, U, b_tag, V, b_t])
        return loss, grads[1]


    def _loss_and_U_grad(U):
        global W
        global b
        global b_tag
        global V
        global b_t
        loss, grads = loss_and_gradients([1, 2, 3], 0, [W, b, U, b_tag, V, b_t])
        return loss, grads[2]


    def _loss_and_b_tag_grad(b_tag):
        global W
        global b
        global U
        global V
        global b_t
        loss, grads = loss_and_gradients([1, 2, 3], 0, [W, b, U, b_tag,V,b_t])
        return loss, grads[3]

    def _loss_and_V_grad(V):
        global W
        global b
        global U
        global b_tag
        global b_t
        loss, grads = loss_and_gradients([1, 2, 3], 0, [W, b, U, b_tag,V,b_t])
        return loss, grads[4]

    def _loss_and_b_t_grad(b_t):
        global W
        global b
        global U
        global b_tag
        global V
        loss, grads = loss_and_gradients([1, 2, 3], 0, [W, b, U, b_tag,V,b_t])
        return loss, grads[5]

    def _loss_and_UU_grad(UU):
        global b
        global U
        global b_tag
        global V
        global b_t
        global W
        global bb
        loss, grads = loss_and_gradients([1, 2, 3], 0, [W, b, U, b_tag,V,b_t,UU,bb])
        return loss, grads[6]

    def _loss_and_bb_grad(bb):
        global b
        global U
        global b_tag
        global V
        global b_t
        global W
        global UU
        loss, grads = loss_and_gradients([1, 2, 3], 0, [W, b, U, b_tag,V,b_t,UU,bb])
        return loss, grads[7]


    for _ in xrange(2):
        print _, '!!!!!!!!!!!!!!!!!!'
        W = np.random.randn(W.shape[0], W.shape[1])
        b = np.random.randn(b.shape[0])
        U = np.random.randn(U.shape[0], U.shape[1])
        b_tag = np.random.randn(b_tag.shape[0])
        V = np.random.randn(V.shape[0], V.shape[1])
        b_t = np.random.randn(b_t.shape[0])
        print 'W:'
        gradient_check(_loss_and_W_grad, W)
        print 'b:'
        gradient_check(_loss_and_b_grad, b)
        print 'U:'
        gradient_check(_loss_and_U_grad, U)
        print 'b_tag:'
        gradient_check(_loss_and_b_tag_grad, b_tag)
        print 'V:'
        gradient_check(_loss_and_V_grad, V)
        print 'bb:'
        gradient_check(_loss_and_b_t_grad, b_t)


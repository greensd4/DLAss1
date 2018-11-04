import numpy as np
import loglinear as ll

STUDENT={'name': 'YOUR NAME',
         'ID': 'YOUR ID NUMBER'}

def classifier_output(x, params):
    p = list(params)
    mlp1_vec = np.dot(p[2],(np.tanh(np.dot(p[0],x)+p[1])))+p[3]
    probs = ll.softmax(mlp1_vec)
    return probs

def predict(x, params):
    """
    params: a list of the form [W, b, U, b_tag]
    """
    return np.argmax(classifier_output(x, params))

def loss_and_gradients(x, y, params):
    """
    params: a list of the form [W, b, U, b_tag]

    returns:
        loss,[gW, gb, gU, gb_tag]

    loss: scalar
    gW: matrix, gradients of W
    gb: vector, gradients of b
    gU: matrix, gradients of U
    gb_tag: vector, gradients of b_tag
    """
    p = list(params)
    probs = classifier_output(x, p)   # probabilities vec
    y_one_hot = np.zeros(len(probs))  # create one-hot vector
    y_one_hot[y] = 1                  # initialize it by the value of y
    W, b, U, b_tag = p
    # gradients values
    gb_tag = -(y_one_hot - probs)
    gU = np.outer(gb_tag, np.tanh(np.dot(W, x) + b))
    gb = np.dot(gb_tag, U) * (1 - np.square(np.tanh(np.dot(W, x) + b)))
    gW = np.outer(gb, x)

    loss = -np.log(probs[y])
    return loss, [gW, gb, gU, gb_tag]

def create_classifier(in_dim, hid_dim, out_dim):
    """
    returns the parameters for a multi-layer perceptron,
    with input dimension in_dim, hidden dimension hid_dim,
    and output dimension out_dim.

    return:
    a flat list of 4 elements, W, b, U, b_tag.
    """

    root_six = np.sqrt(6)
    eps = root_six / (np.sqrt(hid_dim + in_dim))
    W = np.random.uniform(-eps, eps, [hid_dim, in_dim])
    eps = root_six / (np.sqrt(hid_dim))
    b = np.random.uniform(-eps, eps, hid_dim)
    eps = root_six / (np.sqrt(out_dim + hid_dim))
    U = np.random.uniform(-eps, eps, [out_dim, hid_dim])
    eps = root_six / (np.sqrt(out_dim))
    b_tag = np.random.uniform(-eps, eps, out_dim)

    return [W,b,U,b_tag]

if __name__ == '__main__':

    from grad_check import gradient_check

    W,b,U,b_tag = create_classifier(3,3,4)

    def _loss_and_W_grad(W):
        global b
        global U
        global b_tag
        loss,grads = loss_and_gradients([1,2,3],0,[W,b,U,b_tag])
        return loss,grads[0]

    def _loss_and_b_grad(b):
        global W
        global U
        global b_tag
        loss,grads = loss_and_gradients([1,2,3],0,[W,b,U,b_tag])
        return loss,grads[1]

    def _loss_and_U_grad(U):
        global W
        global b
        global b_tag
        loss,grads = loss_and_gradients([1,2,3],0,[W,b,U,b_tag])
        return loss,grads[2]

    def _loss_and_b_tag_grad(b_tag):
        global W
        global b
        global U
        loss,grads = loss_and_gradients([1,2,3],0,[W,b,U,b_tag])
        return loss,grads[3]

    for _ in xrange(10):
        print _, '!!!!!!!!!!!!!!!!!!'
        W = np.random.randn(W.shape[0], W.shape[1])
        b = np.random.randn(b.shape[0])
        U = np.random.randn(W.shape[0], W.shape[1])
        b_tag = np.random.randn(b.shape[0])
        print 'W:'
        gradient_check(_loss_and_W_grad, W)
        print 'b:'
        gradient_check(_loss_and_b_grad, b)
        print 'U:'
        gradient_check(_loss_and_U_grad, U)
        print 'b_tag:'
        gradient_check(_loss_and_b_tag_grad, b_tag)
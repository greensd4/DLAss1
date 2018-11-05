import loglinear as ll
import random
import utils as ut
import numpy as np

STUDENT={'name': 'Daniel Greenspan_Eilon Bashari',
         'ID': '308243948_308576933'}
LR = 0.15
EPOCH = 200


def feats_to_vec(features):
    # Should return a numpy vector of features.
    feats = ut.text_to_bigrams(features)
    feat_vec = np.array(np.zeros(len(ut.F2I)))
    matches = 0
    for bigram in feats:
        if bigram in ut.F2I:
            feat_vec[ut.F2I[bigram]] += 1
            matches += 1
    return np.divide(feat_vec, matches)


def accuracy_on_dataset(dataset, params):
    good = bad = 0.0
    for label, features in dataset:
        # Compute the accuracy (a scalar) of the current parameters
        # on the dataset.
        # accuracy is (correct_predictions / all_predictions)
        pred = ll.predict(feats_to_vec(features), params)
        if pred == ut.L2I[label]:
            good += 1
        else:
            bad += 1
        pass
    return good / (good + bad)


def train_classifier(train_data, dev_data, num_iterations, learning_rate, params):
    """
    Create and train a classifier, and return the parameters.

    train_data: a list of (label, feature) pairs.
    dev_data  : a list of (label, feature) pairs.
    num_iterations: the maximal number of training iterations.
    learning_rate: the learning rate to use.
    params: list of parameters (initial values)
    """
    # costs = []
    # acc = []
    for I in xrange(num_iterations):
        cum_loss = 0.0 # total loss in this iteration.
        random.shuffle(train_data)
        for label, features in train_data:
            x = feats_to_vec(features)         # convert features to a vector.
            y = ut.L2I[label]                  # convert the label to number if needed.
            loss, grads = ll.loss_and_gradients(x, y, params)
            cum_loss += loss
            # update the parameters according to the gradients
            # and the learning rate.
            params[0] -= learning_rate*grads[0]
            params[1] -= learning_rate*grads[1]

        train_loss = cum_loss / len(train_data)
        # costs.append(train_loss)
        train_accuracy = accuracy_on_dataset(train_data, params)
        dev_accuracy = accuracy_on_dataset(dev_data, params)
        # acc.append((train_accuracy,dev_accuracy))
        print I, train_loss, train_accuracy, dev_accuracy
    # fig = plt.plot(acc)
    # fig1 = plt.plot(costs)
    return params


def test(parameters):
    """
    test classifier with test data - no labels

    params - the trained params
    """
    fd = open("test.pred.ll", 'w')
    counter = 0
    test_ans = ''
    test_data = ut.read_data('test')
    for label, feature in test_data:
        pred = ll.predict(feats_to_vec(feature), parameters)
        for l,i in ut.L2I.items():
            if i == pred:
                test_ans = l
        counter += 1
        fd.write(test_ans+"\n")
    fd.close()

if __name__ == '__main__':
    # write code to load the train and dev sets, set up whatever you need,
    # and call train_classifier.

    train_data = ut.read_data('train')
    dev_data = ut.read_data('dev')

    params = ll.create_classifier(len(ut.F2I), len(ut.L2I))
    trained_params = train_classifier(train_data, dev_data, EPOCH, LR, params)
    print 'the final params are:\nW =\n', trained_params[0], '\nb =\n', trained_params[1]
    #test(trained_params)




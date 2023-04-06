from keras import backend as K
import tensorflow as tf

def data_split(matrix, target, test_proportion):
    ratio = int(matrix.shape[0] / test_proportion)
    X_train = matrix[ratio:, :, :]
    X_test = matrix[:ratio, :, :]
    Y_train = matrix[ratio:, :, :]
    Y_test = matrix[:ratio, :, :]
    return X_train, X_test, Y_train, Y_test


def focal_loss(gamma=2., alpha = 0.25):
    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_list(y_pred))
        pt_0 = tf.where(tf.equal(y_pred, 0), y_pred, tf.zeros_like(y_pred))
        return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - K.sum((1-alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0)) 
    return focal_loss_fixed
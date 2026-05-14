from flag_dnn.ops.selu import selu as selu_op


def selu_(input):
    return selu_op(input, inplace=True)

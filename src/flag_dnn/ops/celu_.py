from flag_dnn.ops.celu import celu as celu_op


def celu_(input, alpha: float = 1.0):
    return celu_op(input, alpha=alpha, inplace=True)

import numpy as np


def mac_of_addmm(input1, input2):
    # [n, p] = aten::addmm([n, p], [n, m], [m, p], *, *)
    n, m = input1.shape
    m, p = input2.shape
    return n * m * p / 1e6


def mac_of_addmv(node):
    # [n] = aten::addmv([n], [n, m], [m], *, *)
    n, m = node.inputs[1].shape
    return n * m / 1e6


def mac_of_bmm(node):
    # [b, n, p] = aten::bmm([b, n, m], [b, m, p])
    b, n, m = node.inputs[0].shape
    b, m, p = node.inputs[1].shape
    return b * n * m * p / 1e6


def mac_of_matmul(node):
    if node.inputs[0].ndim == 1 and node.inputs[1].ndim == 1:
        # [] = aten::matmul([n], [n])
        n = node.inputs[0].shape[0]
        return n / 1e6
    elif node.inputs[0].ndim == 1 and node.inputs[1].ndim == 2:
        # [m] = aten::matmul([n], [n, m])
        n, m = node.inputs[1].shape
        return n * m / 1e6
    elif node.inputs[0].ndim == 2 and node.inputs[1].ndim == 1:
        # [n] = aten::matmul([n, m], [m])
        n, m = node.inputs[0].shape
        return n * m / 1e6
    elif node.inputs[0].ndim == 2 and node.inputs[1].ndim == 2:
        # [n, p] = aten::matmul([n, m], [m, p])
        n, m = node.inputs[0].shape
        m, p = node.inputs[1].shape
        return n * m * p / 1e6
    elif node.inputs[0].ndim == 1:
        # [..., m] = aten::matmul([n], [..., n, m])
        *b, n, m = node.inputs[1].shape
        return (np.prod(b) * n * m / 1e6).item()
    elif node.inputs[1].ndim == 1:
        # [..., n] = aten::matmul([..., n, m], [m])
        *b, n, m = node.inputs[0].shape
        return (np.prod(b) * n * m / 1e6).item()
    else:
        # [..., n, p] = aten::matmul([..., n, m], [..., m, p])
        *b, n, p = node.outputs[0].shape
        *_, n, m = node.inputs[0].shape
        *_, m, p = node.inputs[1].shape
        return (np.prod(b) * n * m * p / 1e6).item()


def mac_of_mul(node):
    os = node.outputs[0].shape
    return (np.prod(os) / 1e6).item()


def mac_of_convolution(kweight, input, output):
    if output.shape[1] == kweight.shape[0]:
        oc, ic, *ks = kweight.shape
    else:
        ic, oc, *ks = kweight.shape
    os = output.shape
    return (np.prod(os) * ic * np.prod(ks) / 1e6).item()


def mac_of_batch_norm(node):
    return 0


def mac_of_instance_norm_or_layer_norm(node):
    os = node.outputs[0].shape
    return (np.prod(os) / 1e6).item()


def mac_of_avg_pool_or_mean(output):
    os = output.shape
    return (np.prod(os) / 1e6).item()

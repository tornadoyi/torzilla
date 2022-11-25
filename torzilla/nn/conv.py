import numpy as np

def eval_conv_output_size(conv, input_size):
    '''
    :param conv:  convolution kernel
    :param input_size:  int or tuple  with format size / (H, W) / (D, H, W)
    :return:
    '''
    input_size = np.asarray(input_size)
    kernel_size = np.asarray(conv.kernel_size)
    dilation = np.asarray(conv.dilation)
    padding = np.asarray(conv.padding)
    stride = np.asarray(conv.stride)

    def _conv1d():
        """
        https://pytorch.org/docs/stable/nn.html#conv1d
        """
        numerator = input_size + 2 * padding - dilation * (kernel_size - 1)
        return tuple(np.floor(numerator / stride + 1).astype(np.int))

    def _conv2d():
        """
        https://pytorch.org/docs/stable/nn.html#conv2d
        """
        numerator = input_size + 2 * padding - dilation * (kernel_size - 1) - 1
        return tuple(np.floor(numerator / stride + 1).astype(np.int))

    def _conv3d():
        """
        https://pytorch.org/docs/stable/nn.html#conv3d
        """
        numerator = input_size + 2 * padding - dilation * (kernel_size - 1) - 1
        return tuple(np.floor(numerator / stride + 1).astype(np.int))

    shape = np.shape(conv.kernel_size)
    if len(shape) == 0: return _conv1d()
    elif shape[0] == 2: return _conv2d()
    elif shape[0] == 3: return _conv3d()

    raise Exception('Invalid conv kernel {}'.format(conv.kernel_size))


def eval_pool_output_size(pool, input_size):
    '''
    :param pool:  pool
    :param input_size:  int or tuple  with format size / (H, W) / (D, H, W)
    :return:
    '''
    input_size = np.asarray(input_size)
    kernel_size = np.asarray(pool.kernel_size)
    dilation = np.asarray(pool.dilation)
    padding = np.asarray(pool.padding)
    stride = np.asarray(pool.stride)

    def _pool2d():
        """
        https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html?highlight=maxpool#torch.nn.MaxPool2d
        """
        numerator = input_size + 2 * padding - dilation * (kernel_size - 1) - 1
        return tuple(np.floor(numerator / stride + 1).astype(np.int))

    shape = np.shape(pool.kernel_size)
    if shape[0] == 2: return _pool2d()
    else: raise Exception(f'unsupported kernel dims {len(shape)}')

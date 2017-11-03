from artemis.general.should_be_builtins import memoize
import numpy as np
from six.moves import xrange


__author__ = 'peter'


def vector_length_to_tile_dims(vector_length, ):
    """
    You have vector_length tiles to put in a 2-D grid.  Find the size
    of the grid that best matches the desired aspect ratio.

    TODO: Actually do this with aspect ratio

    :param vector_length:
    :return: n_rows, n_cols
    """
    n_cols = np.ceil(np.sqrt(vector_length))
    n_rows = np.ceil(vector_length/n_cols)
    grid_shape = int(n_rows), int(n_cols)
    return grid_shape


def put_vector_in_grid(vec, shape = None):
    if shape is None:
        n_rows, n_cols = vector_length_to_tile_dims(len(vec))
    else:
        n_rows, n_cols = shape
    grid = np.zeros(n_rows*n_cols, dtype = vec.dtype)
    grid[:len(vec)]=vec
    grid=grid.reshape(n_rows, n_cols)
    return grid


@memoize
def _data_shape_and_boundary_width_to_grid_slices(shape, grid_shape, boundary_width, is_colour = None):

    assert len(shape) in (3, 4) or len(shape)==5 and shape[-1]==3
    if is_colour is None:
        is_colour = shape[-1]==3
    size_y, size_x = (shape[-3], shape[-2]) if is_colour else (shape[-2], shape[-1])
    is_vector = (len(shape)==4 and is_colour) or (len(shape)==3 and not is_colour)

    if grid_shape is None:
        grid_shape = vector_length_to_tile_dims(shape[0]) if is_vector else shape[:2]
    n_rows, n_cols = grid_shape

    output_shape = n_rows*(size_y+boundary_width)+1, n_cols*(size_x+boundary_width)+1
    index_pairs = []
    for i in xrange(n_rows):
        for j in xrange(n_cols):
            if is_vector:
                pull_indices = (i*n_cols + j, )
                if pull_indices[0] == shape[0]:
                    break
            else:
                pull_indices = (i, j)
            start_row, start_col = i*(size_y+1)+1, j*(size_x+1)+1
            push_indices = slice(start_row, start_row+size_y), slice(start_col, start_col+size_x)
            index_pairs.append((pull_indices, push_indices))
    return output_shape, index_pairs


def put_data_in_grid(data, grid_shape = None, fill_colour = np.array((0, 0, 128), dtype = 'uint8'), cmap = 'gray',
        boundary_width = 1, clims = None, is_color_data=None, nan_colour=None):
    """
    Given a 3-d or 4-D array, put it in a 2-d grid.
    :param data: A 4-D array of any data type
    :return: A 3-D uint8 array of shape (n_rows, n_cols, 3)
    """
    output_shape, slice_pairs = _data_shape_and_boundary_width_to_grid_slices(data.shape, grid_shape, boundary_width, is_colour=is_color_data)
    output_data = np.empty(output_shape+(3, ), dtype='uint8')
    output_data[..., :] = fill_colour  # Maybe more efficient just to set the spaces.
    scaled_data = data_to_image(data, clims = clims, cmap = cmap, is_color_data=is_color_data, nan_colour=nan_colour)
    for pull_slice, push_slice in slice_pairs:
        output_data[push_slice] = scaled_data[pull_slice]
    return output_data


def put_list_of_images_in_array(list_of_images, fill_colour = np.array((0, 0, 0))):
    """
    Arrange a list of images into a grid.  They do not necessairlily need to have the same size.

    :param list_of_images: A list of images, not necessarily the same size.
    :param fill_colour: The colour with which to fill the gaps
    :return: A (n_images, size_y, size_x, 3) array of images.
    """
    size_y = max(im.shape[0] for im in list_of_images)
    size_x = max(im.shape[1] for im in list_of_images)
    im_array = np.zeros((len(list_of_images), size_y, size_x, 3))+fill_colour
    for g, im in zip(im_array, list_of_images):
        top = int((size_y-im.shape[0])/2)
        left = int((size_x-im.shape[1])/2)
        g[top:top+im.shape[0], left:left+im.shape[1], :] = im if im.ndim==3 else im[:, :, None]
    return im_array


def put_list_of_lists_of_images_in_array(list_of_lists_of_images, fill_colour = np.array((0, 0, 0))):
    """
    Arrange a list of lists of images into a grid.  Each sublist does not necessarily need to have the same length, and
    images within the lists do not necessairlily need to have the same size.

    :param list_of_lists_of_images: A list of lists of images.
    :param fill_colour: The colour with which to fill the gaps
    :return: A (n_rows, n_cols, size_y, size_x, 3) array of images.
    """
    image_arrays = [put_list_of_images_in_array(list_of_images) for list_of_images in list_of_lists_of_images]
    image_grid_tensor = np.zeros((len(image_arrays), max(arr.shape[0] for arr in image_arrays), max(arr.shape[1] for arr in image_arrays), max(arr.shape[2] for arr in image_arrays), 3))+fill_colour  # (n_rows, n_cols, size_y, size_x, 3)
    for i, arr in enumerate(image_arrays):
        image_grid_tensor[i, :arr.shape[0], :arr.shape[1], :arr.shape[2], :] = arr
    return image_grid_tensor


def scale_data_to_8_bit(data, in_range = None):
    """
    Scale data to range [0, 255] and put in uint8 format.
    """
    return scale_data_to_range(data, in_range=in_range, out_range=(0, 255)).astype(np.uint8)


def scale_data_to_range(data, in_range = None, out_range = (0, 1), clip_to_range = True):
    """
    Scale your data into a new range.

    :param data: An ndarray
    :param in_range: The range that you'd like to scale your data from (if None, it's calculated automatically)
    :param out_range: The range you'd like to scale your data to.
    :param clip_to_range: True if you want to cut data outside of your range.
    :return: An ndarray with values in [out_range[0], out_range[1]]
    """
    out_scale = float(out_range[1]-out_range[0])
    out_shift = out_range[0]

    compute_scale = in_range is None
    if compute_scale:
        in_range = (np.nanmin(data), np.nanmax(data)) if data.size != 0 else (0, 1)
    smin, smax = in_range
    if smin==smax:
        smax += 1.
    scale = out_scale/(smax-smin)

    if np.isnan(scale):  # Data is all nans, or min==max
        return np.zeros_like(data)
    else:
        out = (data-smin)*scale
        if out_shift != 0:
            out += out_shift

        if not compute_scale and clip_to_range:
            out[out<out_range[0]] = out_range[0]
            out[out>out_range[1]] = out_range[1]
        return out


mappables = {}


def data_to_image(data, is_color_data = None, clims = None, cmap = 'gray', nan_colour=None):
    import matplotlib.cm as cm
    from matplotlib.colors import Normalize
    """
    Convert and ndarray of data into RGB pixel data.

    :param data: An ndarray of data.
    :param is_color_data: A boolean indicating whether this is colour data already.  If not specified we guess.
    :param clims: The range of values that the colour scale should cover.  Values outside this range will be
        clipped to fall in the range.  If None, calculate range from the data.
    :param cmap: Colormap - Use any of the names in matplotlib - eg ('gray', 'jet', 'Paired', 'cubehelix')
    :return: An ndarray of unt8 colour data.  Shape is: data.shape if is_color_data else data.shape+(3, )
    """

    if is_color_data is None:
        is_color_data = data.shape[-1] == 3
    if is_color_data:
        assert data.shape[-1] == 3, 'If data is specified as being colour data, the final axis must have length 3.'

    if not is_color_data:
        # Need to apply the cmap.
        if cmap == 'gray':
            # For speed, we handle this separately
            scaled_data = scale_data_to_8_bit(data, in_range=clims)
            scaled_data = np.concatenate([scaled_data[..., None]]*3, axis = scaled_data.ndim)
        else:
            if (clims, cmap) not in mappables:
                mappables[clims, cmap] = cm.ScalarMappable(cmap = cmap, norm = None if clims is None else Normalize(vmin=clims[0], vmax=clims[1]))
            cmap = mappables[clims, cmap]
            old_dim = data.shape
            if len(old_dim)>2:
                data = data.reshape((data.shape[0], -1))
            rgba = cmap.to_rgba(data)
            if len(old_dim)>2:
                rgba = rgba.reshape(old_dim+(4, ))
            scaled_data = (rgba[..., :-1]*255)
    else:
        scaled_data = scale_data_to_8_bit(data, in_range=clims).astype(np.uint8)

    if nan_colour is not None:
        scaled_data = np.where(np.any(np.isnan(data if is_color_data else data[..., None]), axis=-1)[..., None], np.array(nan_colour, dtype=np.uint8), scaled_data)

    return scaled_data


class RecordBuffer(object):

    def __init__(self, buffer_len, initial_value = np.NaN):
        self._buffer_len = buffer_len
        self._buffer = None
        self._ix = 0
        self._base_indices = np.arange(buffer_len)
        self._initial_value = initial_value

    def __call__(self, data):
        if self._buffer is None:
            shape = () if np.isscalar(data) else data.shape
            dtype_data = data+self._initial_value
            dtype = dtype_data.dtype if isinstance(dtype_data, np.ndarray) else type(dtype_data) if isinstance(dtype_data, (int, float, bool)) else object
            self._buffer = np.empty((self._buffer_len, )+shape, dtype = dtype)
            self._buffer[:] = self._initial_value
        self._buffer[self._ix] = data
        self._ix = (self._ix+1) % self._buffer_len
        return self._buffer[(self._base_indices+self._ix) % self._buffer_len]


class UnlimitedRecordBuffer(object):

    def __init__(self, expansion_factor = 2, initial_size=64):
        self._buffer = []
        self._index = 0
        self.initial_size = initial_size
        self.expansion_factor = expansion_factor

    def __call__(self, data):
        if self._index==len(self._buffer):
            data_shape = data.shape if isinstance(data, np.ndarray) else ()
            if len(self._buffer)==0:
                self._buffer = np.empty((self.initial_size, )+data_shape)
            else:
                new_buffer = np.empty((int(len(self._buffer)*self.expansion_factor), )+data_shape)
                new_buffer[:self._index] = self._buffer
                self._buffer = new_buffer
        self._buffer[self._index] = data
        self._index += 1
        return self._buffer[:self._index]


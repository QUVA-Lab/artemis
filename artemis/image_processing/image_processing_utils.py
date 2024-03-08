from typing import Tuple, Optional
import numpy as np
import cv2

from artemis.general.custom_types import Array, HeatMapArray, BGRImageArray, AnyImageArray, BGRFloatImageArray, LabelImageArray
from artemis.plotting.easy_window import ImageRow
from artemis.image_processing.image_utils import heatmap_to_color_image, delta_image_to_color_image
from artemis.plotting.cv2_plotting import just_show


def box_filter(image: AnyImageArray, ksize: Tuple[int, int], normalize: bool = True, n_iter: int = 1):
    result = image
    for _ in range(n_iter):
        result = cv2.boxFilter(result, ddepth=-1, ksize=ksize, normalize=normalize)
    return result


def box_blur(image: AnyImageArray, width: int, normalize: bool = True, weights: Optional[HeatMapArray] = None, n_iter: int = 1) -> AnyImageArray:
    if weights is not None:
        image = image * weights[:, :, None]

    result = box_filter(image, ksize=(width, width), normalize=normalize and weights is None, n_iter=n_iter)

    if normalize and weights is not None:
        weight_sum = box_filter(weights, ksize=(width, width), normalize=False, n_iter=n_iter)
        result /= weight_sum[:, :, None]  # May be nans...

    return result


def approx_gaussian_blur(image: BGRImageArray, sigma: float, weights: Optional[HeatMapArray] = None, n_steps: int = 3) -> BGRImageArray:
    """
    Approximate a gaussian blur with a series of box blurs.
        gaussian_variance: sigma**2
        sum_of_uniform variance: n_steps*kwidth**2/12
        k_width = sqrt( 12 * sigma**2 / n_steps)

    :param image:
    :param sigma:
    :param weights:
    :param n_steps:
    :return:
    """
    kwidth = round(np.sqrt(12 * sigma ** 2 / n_steps))
    if weights is not None:
        image = image * weights[:, :, None]
    result = box_filter(image, ksize=(kwidth, kwidth), n_iter=n_steps)
    if weights is not None:
        weight_sum = box_filter(weights, (kwidth, kwidth), n_iter=n_steps)
        result /= weight_sum[:, :, None]
    return result


def gaussian_blur(image: BGRImageArray, sigma: float, truncation_factor: float = 3., weights: Optional[HeatMapArray] = None) -> BGRImageArray:
    ksize = int(truncation_factor * sigma) * 2 + 1
    if weights is not None:
        image = image * weights[:, :, None]
    result = cv2.GaussianBlur(image, ksize=(ksize, ksize), sigmaX=sigma)
    if weights is not None:
        weight_sum = cv2.GaussianBlur(weights, (ksize, ksize), sigmaX=sigma)
        result /= weight_sum[:, :, None]
    return result


def holy_box_blur(image: AnyImageArray, outer_box_width: int, inner_box_width: int, normalize: bool = True, weights: Optional[HeatMapArray] = None,
                  n_iter: int = 1) -> AnyImageArray:
    # TODO: Be more efficient about use of weights to avoid duplication of computation
    local_sum = box_blur(image, width=inner_box_width, normalize=False, weights=weights, n_iter=n_iter)
    big_box_sum = box_blur(image, width=outer_box_width, normalize=False, weights=weights, n_iter=n_iter).astype(float)
    context_mean = (big_box_sum - local_sum)  # Not actually a mean yet - as we have to normalize still
    if normalize:
        if weights is not None:
            divisor = holy_box_blur(image=weights, normalize=False, n_iter=n_iter, inner_box_width=inner_box_width, outer_box_width=outer_box_width)
            context_mean /= divisor[:, :, None]
        else:
            context_mean /= (outer_box_width ** 2 - inner_box_width ** 2)

    return context_mean


def compute_center_surround_means(heatmap, inner_box_width: int, outer_box_width: int
                                  ) -> Tuple[HeatMapArray, HeatMapArray]:
    center_blur = box_blur(heatmap, width=inner_box_width, normalize=False)
    surround_blur = box_blur(heatmap, width=outer_box_width, normalize=False)
    center_blur /= inner_box_width**2
    surround_blur /= outer_box_width ** 2 - inner_box_width ** 2
    return center_blur, surround_blur

def compute_aloneness_factor(heatmap: HeatMapArray, outer_box_width: int, inner_box_width: int, suppression_factor: float = 1,
                             n_iter: int = 1, debug=False) -> HeatMapArray:
    """ Compute an 'aloneness factor' which indicates how "alone" each region in the heatmap is.  This will b """
    # center_blur = box_blur(heatmap, width=inner_box_width, normalize=False)
    # big_box_sum = box_blur(heatmap, width=outer_box_width, normalize=False)
    # surround_blur = big_box_sum - center_blur
    # center_blur *= center_scale / inner_box_width**2
    # surround_blur *= surround_scale/(outer_box_width ** 2 - inner_box_width ** 2)
    # # suppressed = np.clip(center_blur-surround_blur, 0, None)
    # suppressed = np.exp((surround_blur-center_blur)/surround_blur)

    suppressed = heatmap
    for i in range(n_iter):
        center_blur = box_blur(suppressed, width=inner_box_width, normalize=False)  / inner_box_width**2
        surround_blur = box_blur(suppressed, width=outer_box_width, normalize=False) * suppression_factor / outer_box_width**2
        # suppressed = np.exp((center_blur-surround_blur)/surround_blur)
        suppressed = np.clip(center_blur-surround_blur, 0, None)

        if debug:
            img = ImageRow(heatmap=heatmap_to_color_image(heatmap, show_range=True),
                           center_blur = heatmap_to_color_image(center_blur, show_range=True),
                           surround_blur = heatmap_to_color_image(surround_blur, show_range=True),
                           delta = delta_image_to_color_image(center_blur-surround_blur, show_range=True),
                           suppressed=heatmap_to_color_image(suppressed, show_range=True),
                           wrap=2
                           ).render()
            just_show(img, hang_time=0.1)


    # center_blur = box_blur(heatmap, width=inner_box_width, normalize=False)
    # surround_blur = box_blur(heatmap, width=outer_box_width, normalize=False)
    #
    # suppressed = np.exp( (surround_blur - center_blur) / surround_blur)

    # surround_blur = big_box_sum - center_blur
    # center_blur *= center_scale / inner_box_width**2
    # surround_blur *= surround_scale/(outer_box_width ** 2 - inner_box_width ** 2)
    # suppressed = np.clip(center_blur-surround_blur, 0, None)





    return suppressed  # TODO: Is this right?


def non_maximal_suppression(heatmap: HeatMapArray, outer_box_width: int, inner_box_width: int, suppression_factor: float = 1,
                            n_iter: int = 1) -> HeatMapArray:

    suppression_map = heatmap
    for i in range(n_iter):
        center_mean, surround_mean = compute_center_surround_means(heatmap, inner_box_width=inner_box_width, outer_box_width=outer_box_width)
        surround_mean *= suppression_factor
        relative_diff = (center_mean-surround_mean) / (2*(center_mean + surround_mean))
        suppression_map = np.exp(relative_diff)
    return suppression_map





def compute_context_mean_global_var_background_model(
        image: BGRImageArray,
        anomaly_size: int,
        context_size: int,
        base_variance: float = 0.,
        gaussian_approximation_level: int = 0,
        weights: Optional[HeatMapArray] = None,
        use_context_mean_to_find_var: bool = False
) -> Tuple[Array['H,W,C', float], Array['C,C', float]]:
    """ Compute a pixelwise 'background model' with a position-varying mean and a global covariance.  """
    n_colours = image.shape[2]
    image = image.astype(float, copy=False)
    context_mean = holy_box_blur(image, outer_box_width=context_size, inner_box_width=anomaly_size, weights=weights, n_iter=gaussian_approximation_level + 1)

    if use_context_mean_to_find_var:
        context_covariance = np.cov((image - context_mean).reshape(-1, n_colours), rowvar=False, aweights=weights.ravel() if weights is not None else None)
    else:
        context_covariance = np.cov(image.reshape(-1, n_colours), rowvar=False, aweights=weights.ravel() if weights is not None else None)

    ixs = np.arange(n_colours)
    context_covariance[ixs, ixs] = np.maximum(context_covariance[ixs, ixs], base_variance)
    return context_mean, context_covariance


def compute_pixel_mean_and_cov(image: BGRImageArray) -> Tuple[Array['3', float], Array['3,3', float]]:
    # TODO: Speed up by using mean for cov
    image = image.reshape(-1, image.shape[2])
    mean = image.mean(axis=0)
    cov = np.cov(image)
    return mean, cov


def compute_pixelwise_mahalanobis_dist_sq(
        image: BGRImageArray,
        model_mean: BGRFloatImageArray,
        model_cov: Array['...,3,3', float]
) -> HeatMapArray:
    delta_from_mean = (image - model_mean).reshape(-1, 3)
    m_dist_sq = np.einsum('ij,jk,ik->i', delta_from_mean, np.linalg.inv(model_cov), delta_from_mean).reshape(image.shape[:2])
    return m_dist_sq


def compute_cluster_mean_holy_image(image: BGRImageArray, clusters: LabelImageArray, inner_width: int, outer_width: int) -> BGRImageArray:
    """ Compute the image taken """
    cluster_ids = np.unique(clusters)

    output_image = np.empty_like(image)
    for c in cluster_ids:
        mask = clusters == c
        avg = holy_box_blur(image, outer_box_width=outer_width, inner_box_width=inner_width, weights=mask.astype(np.float32))
        output_image[mask] = avg[mask]
    return output_image





import numpy as ncp
import scipy.ndimage as ndimage

cupy_used = False


def phong_shading_with_background(background: ncp.ndarray, depth_map: ncp.ndarray, led_directions: ncp.ndarray,
                                  specular_components: ncp.ndarray, diffuse_components: ncp.ndarray, alpha: float,
                                  kd: float, ks: float) -> ncp.ndarray:
    """
    Compue the Phong shading of the given depth map
    :param background:          Image to put into the background
    :param depth_map:           Depth map to compute shading for
    :param led_directions:      Directions of the incoming LED light (Nx3 array)
    :param specular_components: Specular components of each LED (Nx3 array)
    :param diffuse_components:  Diffuse components of each LED (Nx3 array)
    :param alpha:               Alpha parameter
    :param kd:                  kd parameter
    :param ks:                  ks parameter
    :return: Phong shaded image
    """
    derivative_kernel_x = ncp.array([[1, 0, -1]])
    derivative_kernel_y = derivative_kernel_x.T

    depth_map_x_derivative = ndimage.convolve(depth_map, derivative_kernel_x, mode="nearest")
    depth_map_y_derivative = ndimage.convolve(depth_map, derivative_kernel_y, mode="nearest")

    V = ncp.array([0, 0, -1])
    height, width, _ = background.shape
    N = ncp.stack([depth_map_x_derivative,
                   depth_map_y_derivative,
                   -ncp.ones_like(depth_map_x_derivative)], axis=-1)
    N /= ncp.linalg.norm(N, axis=-1)[:, :, ncp.newaxis]

    corr = ncp.maximum(0.0, N.dot(led_directions.T))
    R = 2 * corr[:, :, :, ncp.newaxis] * N[:, :, ncp.newaxis, :] - led_directions[ncp.newaxis, ncp.newaxis, :, :]
    R /= ncp.linalg.norm(R, axis=-1)[:, :, :, ncp.newaxis]

    result = kd * corr[:, :, :, ncp.newaxis] * diffuse_components[ncp.newaxis, ncp.newaxis, :, :] \
             + ks * ncp.maximum(0.0, R.dot(V))[:, :, :, ncp.newaxis] ** alpha \
             * specular_components[ncp.newaxis, ncp.newaxis, :, :]
    result = ncp.sum(result, axis=2)

    result += background

    return result

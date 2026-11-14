import base64
import numpy as np
from scipy import ndimage as ndi
from PIL import Image
from .imaging import disk
from .rendering import png_bytes
from .validation import ValidationError


CHANNEL_NAMES = ('red', 'green', 'blue')
BAD_PIXEL_FLOOR = 8.0
PREVIEW_PIXEL_LIMIT = 4000000


def rgb_array(image):
    return np.asarray(image.convert('RGB'), dtype=np.float64)


def data_url(image):
    return 'data:image/png;base64,' + base64.b64encode(png_bytes(image)).decode('ascii')


def channel_defects(channel, sigma_limit):
    """Flag isolated hot or dead pixels, not edges of real structures."""
    neighbourhood = np.ones((3, 3), dtype=bool)
    neighbourhood[1, 1] = False
    brightest = ndi.maximum_filter(channel, footprint=neighbourhood, mode='reflect')
    darkest = ndi.minimum_filter(channel, footprint=neighbourhood, mode='reflect')
    local = ndi.median_filter(channel, size=3, mode='reflect')
    residual = channel - local
    center = float(np.median(residual))
    spread = float(np.median(np.abs(residual - center))) * 1.4826
    threshold = max(sigma_limit * spread, BAD_PIXEL_FLOOR)
    hot = channel > brightest + threshold
    dead = channel < darkest - threshold
    return (hot | dead) & (np.abs(residual - center) > threshold)


def mark_unreliable(image, unreliable):
    base = np.asarray(image.convert('RGB'), dtype=np.float64)
    base[unreliable] = base[unreliable] * 0.35 + np.array([255.0, 46.0, 214.0]) * 0.65
    return Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))


def mask_image(mask):
    """White marks the flagged region so the PNG reads naturally on its own."""
    return Image.fromarray((np.asarray(mask, dtype=bool).astype(np.uint8) * 255))


def preview_image(image, unreliable):
    if image.width * image.height <= PREVIEW_PIXEL_LIMIT:
        return mark_unreliable(image, unreliable)
    copy = image.copy()
    copy.thumbnail((2000, 2000), Image.Resampling.LANCZOS)
    mask = mask_image(unreliable)
    mask.thumbnail(copy.size, Image.Resampling.BOX)
    return mark_unreliable(copy, np.asarray(mask) > 127)


def dilate_mask(mask, margin):
    return ndi.binary_dilation(mask, structure=disk(margin)) if margin else mask


def correct_flatfield(sample, flat, dark, parameters, inherited=None):
    """Per-channel flat/dark correction with explicit unreliable regions.

    The reference response (flat - dark) is normalised per channel, so the
    relative detector response of red, green and blue is kept instead of being
    forced to a common white level. Pixels with defects, saturation or a
    response too small to divide by are marked unreliable rather than used.
    """
    sample_array = rgb_array(sample)
    flat_array = rgb_array(flat)
    dark_array = rgb_array(dark)
    if sample_array.shape != flat_array.shape or sample_array.shape != dark_array.shape:
        raise ValidationError('样本、明场与暗场参考必须尺寸一致')
    height, width = sample_array.shape[:2]
    inherited_mask = None
    if inherited is not None:
        inherited_mask = np.asarray(inherited, dtype=bool)
        if inherited_mask.shape != (height, width):
            raise ValidationError('已有可靠区域掩膜与当前图像尺寸不一致')
    valid = np.ones((height, width), dtype=bool)
    if inherited_mask is not None:
        valid &= inherited_mask
    saturation = parameters['saturation_level']
    response = flat_array - dark_array
    signal = sample_array - dark_array
    corrected = np.zeros_like(sample_array)
    unreliable = np.zeros((height, width), dtype=bool)
    channels = []
    clipped = np.zeros((height, width), dtype=bool)
    for index, name in enumerate(CHANNEL_NAMES):
        defects = channel_defects(flat_array[:, :, index], parameters['bad_pixel_sigma'])
        defects |= channel_defects(dark_array[:, :, index], parameters['bad_pixel_sigma'])
        saturated = (flat_array[:, :, index] >= saturation) | (dark_array[:, :, index] >= saturation)
        channel_response = response[:, :, index]
        usable = ~defects & ~saturated & (channel_response > 0)
        positive = channel_response[usable]
        if positive.size == 0:
            raise ValidationError(f'{name}通道没有有效响应，请检查明场与暗场参考')
        median_response = float(np.median(positive))
        if median_response <= 0:
            raise ValidationError(f'{name}通道中位响应为零，无法校正')
        gain_limit = median_response / parameters['max_gain']
        low_limit = max(
            parameters['min_response'],
            parameters['min_response_fraction'] * median_response,
            gain_limit,
        )
        low_response = channel_response < low_limit
        gain = median_response / np.maximum(channel_response, 1e-9)
        scaled = signal[:, :, index] * gain
        # A hot or dead detector pixel in the sample itself shows up after
        # scaling; flag it against its local median so division-amplified
        # speckles are not presented as structure.
        sample_defects = channel_defects(scaled, parameters['bad_pixel_sigma'])
        channel_unreliable = defects | saturated | low_response | sample_defects
        channel_valid = valid & ~channel_unreliable & np.isfinite(scaled)
        fill = float(np.median(scaled[channel_valid])) if np.any(channel_valid) else 0.0
        scaled = np.where(channel_valid, scaled, fill)
        clipped |= (scaled < 0) | (scaled > 255)
        corrected[:, :, index] = np.clip(scaled, 0, 255)
        unreliable |= channel_unreliable
        channels.append({
            'channel': name,
            'median_dark': round(float(np.median(dark_array[:, :, index])), 6),
            'median_flat': round(float(np.median(flat_array[:, :, index])), 6),
            'median_response': round(median_response, 6),
            'low_response_limit': round(float(low_limit), 6),
            'gain_limit_response': round(float(gain_limit), 6),
            'bad_pixels': int(np.count_nonzero(defects)),
            'sample_bad_pixels': int(np.count_nonzero(sample_defects)),
            'saturated_pixels': int(np.count_nonzero(saturated)),
            'low_response_pixels': int(np.count_nonzero(low_response)),
            'unreliable_pixels': int(np.count_nonzero(channel_unreliable)),
        })
    final_unreliable = dilate_mask(unreliable, parameters['margin'])
    if inherited_mask is not None:
        final_unreliable |= ~inherited_mask
    total = int(final_unreliable.size)
    unreliable_pixels = int(np.count_nonzero(final_unreliable))
    image = Image.fromarray(np.clip(corrected, 0, 255).astype(np.uint8))
    reliable = ~final_unreliable
    details = {
        'width': width,
        'height': height,
        'parameters': {
            'saturation_level': parameters['saturation_level'],
            'min_response': parameters['min_response'],
            'min_response_fraction': parameters['min_response_fraction'],
            'max_gain': parameters['max_gain'],
            'bad_pixel_sigma': parameters['bad_pixel_sigma'],
            'margin': parameters['margin'],
        },
        'channel_names': list(CHANNEL_NAMES),
        'channels': channels,
        'clipped_pixels': int(np.count_nonzero(clipped & reliable)),
        'unreliable_pixels': unreliable_pixels,
        'unreliable_percent': round(unreliable_pixels / max(total, 1) * 100, 6),
        'reliable_pixels': total - unreliable_pixels,
        'reliable_percent': round((total - unreliable_pixels) / max(total, 1) * 100, 6),
        'inherited_unreliable_pixels': int(np.count_nonzero(~inherited_mask)) if inherited_mask is not None else 0,
        'preview_downscaled': image.width * image.height > PREVIEW_PIXEL_LIMIT,
        'formula': '校正 = (样本 − 暗场) × 该通道参考响应中位数 / (明场 − 暗场)',
        'warning': '坏点、饱和、响应过低或增益过高的像素已标为不可靠区域；洋红为不可靠区域，另存后的新样本分析会排除这些像素，不把除法放大的斑点当作结构',
    }
    return {
        'image': image,
        'reliable': reliable,
        'unreliable': final_unreliable,
        'preview': preview_image(image, final_unreliable),
        'unreliable_mask': mask_image(final_unreliable),
        'details': details,
    }

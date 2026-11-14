import math
import numpy as np
from scipy import ndimage as ndi
from PIL import Image, ImageOps, ImageEnhance, ImageFilter, ImageDraw
from .validation import ValidationError, rectangle


def channel_array(image, channel='gray'):
    rgb = np.asarray(image.convert('RGB'), dtype=np.float32)
    if channel == 'gray':
        return np.asarray(image.convert('L'), dtype=np.float32)
    if channel in ('red', 'green', 'blue'):
        return rgb[:, :, {'red': 0, 'green': 1, 'blue': 2}[channel]]
    hsv = np.asarray(image.convert('HSV'), dtype=np.float32)
    if channel in ('hue', 'saturation', 'value'):
        return hsv[:, :, {'hue': 0, 'saturation': 1, 'value': 2}[channel]]
    raise ValidationError('不支持的颜色通道')


def byte_image(array):
    return Image.fromarray(np.clip(array, 0, 255).astype(np.uint8))


def disk(radius):
    y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    return x * x + y * y <= radius * radius


def normalize_array(array, low_percentile=1, high_percentile=99):
    low, high = np.percentile(array, [low_percentile, high_percentile])
    if high <= low:
        return array.copy()
    return np.clip((array - low) * 255 / (high - low), 0, 255)


def gaussian(image, sigma):
    array = np.asarray(image, dtype=np.float32)
    smooth = ndi.gaussian_filter(array, (sigma, sigma, 0), mode='reflect')
    return byte_image(smooth)


def median(image, radius):
    size = min(radius * 2 + 1, 31)
    return image.filter(ImageFilter.MedianFilter(size))


def bilateral(image, radius, amount):
    rgb = np.asarray(image, dtype=np.float32)
    radius = min(radius, 5)
    padded = np.pad(rgb, ((radius, radius), (radius, radius), (0, 0)), mode='reflect')
    output = np.zeros_like(rgb)
    weight_sum = np.zeros(rgb.shape[:2], dtype=np.float32)
    spatial_sigma = max(radius / 2, 0.5)
    color_sigma = max(amount * 20, 1)
    for oy in range(-radius, radius + 1):
        for ox in range(-radius, radius + 1):
            shifted = padded[
                radius + oy:radius + oy + rgb.shape[0],
                radius + ox:radius + ox + rgb.shape[1],
            ]
            distance = np.mean((shifted - rgb) ** 2, axis=2)
            spatial = math.exp(-(ox * ox + oy * oy) / (2 * spatial_sigma ** 2))
            weight = spatial * np.exp(-distance / (2 * color_sigma ** 2))
            output += shifted * weight[:, :, None]
            weight_sum += weight
    output /= np.maximum(weight_sum[:, :, None], 1e-9)
    return byte_image(output)


def background_correct(image, radius, amount):
    array = np.asarray(image, dtype=np.float32)
    background = ndi.gaussian_filter(array, (radius, radius, 0), mode='reflect')
    center = np.mean(background, axis=(0, 1), keepdims=True)
    corrected = array - amount * (background - center)
    return byte_image(corrected)


def clahe(image, tile_size=64, clip_limit=2):
    gray = np.asarray(image.convert('L'), dtype=np.uint8)
    height, width = gray.shape
    ny = math.ceil(height / tile_size)
    nx = math.ceil(width / tile_size)
    tables = np.zeros((ny, nx, 256), dtype=np.float32)
    for iy in range(ny):
        for ix in range(nx):
            tile = gray[
                iy * tile_size:min((iy + 1) * tile_size, height),
                ix * tile_size:min((ix + 1) * tile_size, width),
            ]
            hist = np.bincount(tile.ravel(), minlength=256).astype(float)
            limit = max(1, clip_limit * tile.size / 256)
            excess = np.maximum(hist - limit, 0).sum()
            hist = np.minimum(hist, limit)
            hist += excess / 256
            cumulative = np.cumsum(hist)
            tables[iy, ix] = cumulative * 255 / max(cumulative[-1], 1)
    yy, xx = np.indices(gray.shape, dtype=np.float32)
    fy = np.clip(yy / tile_size - 0.5, 0, ny - 1)
    fx = np.clip(xx / tile_size - 0.5, 0, nx - 1)
    y0 = np.floor(fy).astype(int)
    x0 = np.floor(fx).astype(int)
    y1 = np.minimum(y0 + 1, ny - 1)
    x1 = np.minimum(x0 + 1, nx - 1)
    wy = fy - y0
    wx = fx - x0
    mapped = (
        tables[y0, x0, gray] * (1 - wx) * (1 - wy)
        + tables[y0, x1, gray] * wx * (1 - wy)
        + tables[y1, x0, gray] * (1 - wx) * wy
        + tables[y1, x1, gray] * wx * wy
    )
    return byte_image(mapped).convert('RGB')


def apply_operation(image, parameters):
    operation = parameters['operation']
    amount = parameters['amount']
    radius = parameters['radius']
    if operation == 'gaussian':
        return gaussian(image, min(amount, 8))
    if operation == 'median':
        return median(image, min(radius, 15))
    if operation == 'bilateral':
        if image.width * image.height > 4000000:
            raise ValidationError('双边滤波限400万像素，请先裁切或缩放')
        return bilateral(image, radius, amount)
    if operation == 'normalize':
        rgb = np.asarray(image, dtype=np.float32)
        output = np.stack([normalize_array(rgb[:, :, c]) for c in range(3)], axis=2)
        return byte_image(output)
    if operation == 'gamma':
        array = np.asarray(image, dtype=np.float32) / 255
        return byte_image(255 * np.power(array, amount))
    if operation == 'background':
        return background_correct(image, radius, min(amount, 3))
    if operation == 'unsharp':
        return image.filter(ImageFilter.UnsharpMask(radius=min(radius, 20), percent=int(amount * 100), threshold=2))
    if operation == 'invert':
        return ImageOps.invert(image)
    if operation == 'channel':
        return byte_image(channel_array(image, parameters['channel'])).convert('RGB')
    if operation == 'contrast':
        return ImageEnhance.Contrast(image).enhance(amount)
    if operation == 'equalize':
        return ImageOps.equalize(image)
    if operation == 'clahe':
        return clahe(image, max(16, radius * 8), min(amount, 10))
    if operation == 'rotate':
        return image.rotate(parameters['angle'], resample=Image.Resampling.BICUBIC, expand=True, fillcolor='white')
    if operation == 'flip':
        if parameters['direction'] == 'horizontal':
            return ImageOps.mirror(image)
        return ImageOps.flip(image)
    if operation == 'crop':
        rect = rectangle(parameters['crop'] or {}, image.width, image.height)
        if min(rect['width'], rect['height']) < 16:
            raise ValidationError('裁切结果至少16×16')
        return image.crop((rect['x'], rect['y'], rect['x'] + rect['width'], rect['y'] + rect['height']))
    if operation == 'resize':
        size = (parameters['width'], parameters['height'])
        if size[0] * size[1] > 16000000:
            raise ValidationError('缩放结果不超过1600万像素')
        return image.resize(size, Image.Resampling.LANCZOS)
    raise ValidationError('不支持的图像操作')


def region_mask(size, region, reliable=None):
    if isinstance(region, np.ndarray):
        selected = np.asarray(region, dtype=bool)
        if selected.shape != (size[1], size[0]):
            raise ValidationError('区域掩膜尺寸与图像不一致')
        return selected if reliable is None else selected & np.asarray(reliable, dtype=bool)
    image = Image.new('1', size, 0)
    draw = ImageDraw.Draw(image)
    kind = region['type']
    if kind == 'full':
        selected = np.ones((size[1], size[0]), dtype=bool)
    elif kind in ('rectangle', 'ellipse'):
        x, y = region['x'], region['y']
        box = (x, y, x + region['width'] - 1, y + region['height'] - 1)
        if kind == 'ellipse':
            draw.ellipse(box, fill=1)
        else:
            draw.rectangle(box, fill=1)
        selected = np.asarray(image, dtype=bool)
    else:
        draw.polygon([tuple(p) for p in region['points']], fill=1)
        selected = np.asarray(image, dtype=bool)
    if reliable is not None:
        selected = selected & np.asarray(reliable, dtype=bool)
    return selected


def intensity_profile(image, start, end, channel='gray', samples=200, line_width=1):
    array = channel_array(image, channel)
    count = max(2, min(int(samples), 2000))
    x = np.linspace(start[0], end[0], count)
    y = np.linspace(start[1], end[1], count)
    length = math.hypot(end[0] - start[0], end[1] - start[1])
    if length == 0:
        raise ValidationError('剖面两端不能重合')
    normal_x = -(end[1] - start[1]) / length
    normal_y = (end[0] - start[0]) / length
    profiles = []
    for offset in np.linspace(-(line_width - 1) / 2, (line_width - 1) / 2, line_width):
        values = ndi.map_coordinates(array, [y + normal_y * offset, x + normal_x * offset], order=1, mode='nearest')
        profiles.append(values)
    mean = np.mean(profiles, axis=0)
    return {
        'distance_px': np.linspace(0, length, count).round(4).tolist(),
        'intensity': mean.round(4).tolist(),
        'length_px': round(length, 4),
        'minimum': round(float(mean.min()), 4),
        'maximum': round(float(mean.max()), 4),
        'mean': round(float(mean.mean()), 4),
        'channel': channel,
        'width': line_width,
    }


def image_histogram(image, region=None, reliable=None):
    if region is None:
        selected = np.ones((image.height, image.width), dtype=bool)
    else:
        selected = region_mask(image.size, region)
    if reliable is not None:
        selected = selected & np.asarray(reliable, dtype=bool)
    result = {}
    for name in ['gray', 'red', 'green', 'blue']:
        values = channel_array(image, name)[selected].astype(np.uint8)
        hist = np.bincount(values, minlength=256)
        result[name] = {
            'bins': hist.tolist(),
            'mean': round(float(values.mean()), 4) if values.size else 0,
            'std': round(float(values.std()), 4) if values.size else 0,
            'minimum': int(values.min()) if values.size else 0,
            'maximum': int(values.max()) if values.size else 0,
            'median': float(np.median(values)) if values.size else 0,
            'pixel_count': int(values.size),
        }
    return result


def quality_metrics(image):
    gray = channel_array(image)
    laplace = ndi.laplace(gray)
    gradient_x = ndi.sobel(gray, axis=1) / 8
    gradient_y = ndi.sobel(gray, axis=0) / 8
    hist = np.bincount(gray.astype(np.uint8).ravel(), minlength=256).astype(float)
    probability = hist / max(hist.sum(), 1)
    positive = probability[probability > 0]
    low = float(np.mean(gray <= 5) * 100)
    high = float(np.mean(gray >= 250) * 100)
    block = ndi.uniform_filter(gray, size=min(101, max(3, min(gray.shape) // 4)))
    warnings = []
    if gray.std() < 12:
        warnings.append('对比度较低，建议先增强对比或调整照明')
    if low > 10:
        warnings.append('暗部像素占比较高，请检查欠曝光')
    if high > 10:
        warnings.append('亮部像素占比较高，请检查过曝光')
    if float(laplace.var()) < 20:
        warnings.append('高频细节较少，可能存在模糊，也可能是样本本身纹理少')
    return {
        'width': image.width,
        'height': image.height,
        'mean': round(float(gray.mean()), 4),
        'contrast_std': round(float(gray.std()), 4),
        'laplacian_variance': round(float(laplace.var()), 4),
        'gradient_mean': round(float(np.hypot(gradient_x, gradient_y).mean()), 4),
        'entropy_bits': round(float(-np.sum(positive * np.log2(positive))), 4),
        'dark_clipping_percent': round(low, 4),
        'bright_clipping_percent': round(high, 4),
        'illumination_span': round(float(np.percentile(block, 95) - np.percentile(block, 5)), 4),
        'warnings': warnings,
    }


def checkerboard(first, second, tile_size=32):
    if first.size != second.size:
        raise ValidationError('棋盘对比要求图像尺寸一致')
    a = np.asarray(first.convert('RGB'))
    b = np.asarray(second.convert('RGB'))
    y, x = np.indices(a.shape[:2])
    selection = ((x // tile_size + y // tile_size) % 2).astype(bool)
    result = np.where(selection[:, :, None], a, b)
    return Image.fromarray(result)


def difference_image(first, second, gain=2):
    if first.size != second.size:
        raise ValidationError('差异图要求图像尺寸一致')
    a = np.asarray(first.convert('RGB'), dtype=float)
    b = np.asarray(second.convert('RGB'), dtype=float)
    diff = np.abs(a - b)
    heat = np.mean(diff, axis=2)
    scaled = np.clip(heat * gain, 0, 255)
    result = np.stack([scaled, np.minimum(scaled * 0.6, 100), 255 - scaled], axis=2)
    return byte_image(result), {
        'mae': round(float(diff.mean()), 6),
        'rmse': round(float(np.sqrt(np.mean((a - b) ** 2))), 6),
        'changed_percent': round(float(np.mean(np.any(diff > 0, axis=2)) * 100), 4),
    }

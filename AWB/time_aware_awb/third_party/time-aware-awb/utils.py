from PIL import Image
from PIL.ExifTags import TAGS
from astral.sun import sun
from astral import LocationInfo
from datetime import datetime, timedelta
import pytz
import math
from typing import List, Union, Dict, Any, Optional, Tuple
from tabulate import tabulate
from timezonefinder import TimezoneFinder
from fractions import Fraction
from colour_demosaicing import demosaicing_CFA_Bayer_Menon2007
import rawpy
import exifread
import torch
import numpy as np
import os
import json
import cv2
import itertools
from constants import *

def dms_to_decimal(degrees: float, minutes: float, seconds: float, direction: str) -> float:
    sign = 1 if direction in ['N', 'E'] else -1
    return sign * (degrees + minutes / 60 + seconds / 3600)

def exposure_time_to_float(exposure_time: Fraction) -> float:
    return exposure_time.numerator / exposure_time.denominator

def shutter_speed_in_seconds(shutter_speed: Fraction) -> float:
    return 1 / (shutter_speed.numerator / shutter_speed.denominator)

def tag_code_to_name(code: int) -> str:
    return TAGS.get(code, f'Tag code {code}')

def get_default_lat_lon():
    return (Fraction('43641903/1000000'), Fraction('-79386707/1000000'))

def processes_exif_metadata(exif_data_: Dict[str, Any], lat: Optional[Fraction]=None, lon: Optional[Fraction]=None) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    exif_data = dict()
    if not hasattr(exif_data_, 'items'):
        raise ValueError('Invalid JPEG file.')
    for tag, value in exif_data_.items():
        tag_name = tag_code_to_name(tag)
        exif_data.update({tag_name: value})
    try:
        gps_info = exif_data['GPSInfo']
        lat = dms_to_decimal(gps_info[2][0], gps_info[2][1], gps_info[2][2], gps_info[1])
        lon = dms_to_decimal(gps_info[4][0], gps_info[4][1], gps_info[4][2], gps_info[3])
    except:
        pass
    if lat is None or lon is None or math.isnan(lat) or math.isnan(lon):
        lat, lon = get_default_lat_lon()
    datetime_original = exif_data['DateTimeOriginal']
    capture_time = datetime.strptime(datetime_original, '%Y:%m:%d %H:%M:%S')
    timezone_finder = TimezoneFinder()
    time_zone_str = timezone_finder.timezone_at(lng=lon, lat=lat)
    time_zone = pytz.timezone(time_zone_str)
    localized_capture_time = time_zone.localize(capture_time)
    capturing_hour = localized_capture_time.hour
    location = LocationInfo(latitude=lat, longitude=lon, timezone=time_zone)
    s = sun(location.observer, date=localized_capture_time)
    diff_sunrise = s['sunrise'].astimezone(time_zone) - localized_capture_time
    time_difference_sunrise = abs(diff_sunrise.total_seconds())
    diff_sunset = s['sunset'].astimezone(time_zone) - localized_capture_time
    time_difference_sunset = abs(diff_sunset.total_seconds())
    diff_dusk = s['dusk'].astimezone(time_zone) - localized_capture_time
    time_difference_dusk = abs(diff_dusk.total_seconds())
    diff_dawn = s['dawn'].astimezone(time_zone) - localized_capture_time
    time_difference_dawn = abs(diff_dawn.total_seconds())
    diff_noon = s['noon'].astimezone(time_zone) - localized_capture_time
    time_difference_noon = abs(diff_noon.total_seconds())
    midnight = localized_capture_time.replace(hour=23, minute=59, second=59, microsecond=0)
    diff_midnight = midnight - localized_capture_time
    time_difference_midnight = abs(diff_midnight.total_seconds())
    prob_midnight = 1 - time_difference_midnight / TOTAL_SECONDS
    prob_sunrise = 1 - time_difference_sunrise / TOTAL_SECONDS
    prob_sunset = 1 - time_difference_sunset / TOTAL_SECONDS
    prob_dusk = 1 - time_difference_dusk / TOTAL_SECONDS
    prob_dawn = 1 - time_difference_dawn / TOTAL_SECONDS
    prob_noon = 1 - time_difference_noon / TOTAL_SECONDS
    is_before_sunrise = 1 if diff_sunrise >= timedelta(0) else 0
    is_before_sunset = 1 if diff_sunset >= timedelta(0) else 0
    is_before_dusk = 1 if diff_dusk >= timedelta(0) else 0
    is_before_dawn = 1 if diff_dawn >= timedelta(0) else 0
    is_before_noon = 1 if diff_noon >= timedelta(0) else 0
    is_before_midnight = 1 if localized_capture_time < midnight else 0
    iso_speed = exif_data.get('ISOSpeedRatings', None)
    exposure_time = exif_data.get('ExposureTime', None)
    if exposure_time is not None:
        exposure_time = exposure_time_to_float(exposure_time)
    exposure_bias = exif_data.get('ExposureBiasValue', None)
    if exposure_bias is not None:
        exposure_bias = exposure_bias.numerator / exposure_bias.denominator
    shutter_speed = shutter_speed_in_seconds(exif_data.get('ShutterSpeedValue', None))
    flash = exif_data.get('Flash', None) & 1
    metadata = {'iso': iso_speed, 'exposure_time': exposure_time, 'exposure_bias': exposure_bias, 'shutter_speed': shutter_speed, 'flash': flash, 'prob_sunrise': prob_sunrise, 'prob_sunset': prob_sunset, 'prob_dusk': prob_dusk, 'prob_dawn': prob_dawn, 'prob_noon': prob_noon, 'prob_midnight': prob_midnight, 'is_before_midnight': is_before_midnight, 'is_before_sunrise': is_before_sunrise, 'is_before_sunset': is_before_sunset, 'is_before_dusk': is_before_dusk, 'is_before_dawn': is_before_dawn, 'is_before_noon': is_before_noon, 'capturing_hour': capturing_hour}
    return (metadata, {'lat': lat, 'lon': lon}, {'localized_capture_time': str(localized_capture_time), 'timezone': time_zone.zone})

def extract_jpg_metadata(image_path: str, pre_lat_lon: Optional[Dict]=None) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    if pre_lat_lon is None:
        lat = lon = None
    else:
        lat = pre_lat_lon['lat']
        lon = pre_lat_lon['lon']
    img = Image.open(image_path)
    exif_data = img._getexif()
    return processes_exif_metadata(exif_data, lat=lat, lon=lon)

def normalize_raw(img: np.ndarray, black_level: Union[List, np.ndarray], white_level: float, cfa_pattern: Optional[str]='rggb') -> np.ndarray:
    raw = img.astype(np.float32)
    if raw.shape[-1] == 4:
        raw = (raw - black_level) / white_level
    else:
        cfa_pattern = cfa_pattern.lower()
        pattern_map = {'rggb': [(0, 0), (0, 1), (1, 0), (1, 1)], 'bggr': [(1, 1), (1, 0), (0, 1), (0, 0)], 'grbg': [(0, 1), (0, 0), (1, 1), (1, 0)], 'gbrg': [(1, 0), (1, 1), (0, 0), (0, 1)]}
        if cfa_pattern not in pattern_map:
            raise ValueError(f'Unsupported Bayer pattern: {cfa_pattern}')
        idx = pattern_map[cfa_pattern]
        height, width = raw.shape
        for i in range(4):
            row, col = idx[i]
            raw[row:height:2, col:width:2] = (raw[row:height:2, col:width:2] - black_level[i]) / white_level
    return raw

def demosaice(img: np.ndarray, cfa_pattern: str) -> np.ndarray:
    if not (img.shape[-1] == 4 and len(img.shape) == 3):
        try:
            cfa_pattern = cfa_pattern.upper()
            img_rgb = demosaicing_CFA_Bayer_Menon2007(img, cfa_pattern)
            return clip(img_rgb)
        except:
            raise NotImplementedError('Unsupported bayer pattern.')
    else:
        img = img[..., :3]
    return img

def clip(x: np.ndarray, min_v: Optional[float]=0.0, max_v: Optional[float]=1.0) -> np.ndarray:
    return np.clip(x, a_min=min_v, a_max=max_v).astype(np.float32)

def rgb_to_rgbg(rgb: np.ndarray) -> np.ndarray:
    return np.stack([rgb[..., 0] / (rgb[..., 1] + EPS), rgb[..., 2] / (rgb[..., 1] + EPS)], axis=-1)

def extract_raw_metadata(file: str) -> Dict[str, Any]:

    def wb_gains_to_illum_color(wb_gains: Union[List, np.ndarray]) -> np.ndarray:
        if not isinstance(wb_gains, np.ndarray):
            wb_gains = np.array(wb_gains)
        wb_gains = wb_gains.flatten()
        if len(wb_gains) == 4:
            if wb_gains[3] == 0:
                wb_gains = wb_gains[:-1]
            else:
                wb_gains[1] = wb_gains[1] + wb_gains[2]
                wb_gains[2] = wb_gains[3]
                wb_gains = wb_gains[:-1]
        illum_color = 1 / np.fmax(wb_gains, EPS)
        return illum_color / np.fmax(np.linalg.norm(illum_color), EPS)

    def process_color_matrix(ccm: np.ndarray) -> np.ndarray:
        if not isinstance(ccm, np.ndarray):
            ccm = np.array(ccm)
        if len(ccm.shape) == 1:
            ccm = ccm.reshape(3, -1)
        if ccm.size > 9:
            ccm = ccm[..., :-1]
        return ccm

    def get_pattern(pattern: str, raw_pattern: Optional[Union[List, np.ndarray]]=None) -> str:
        if raw_pattern is None:
            return pattern
        return ''.join([pattern[i] for i in np.array(raw_pattern).flatten()])

    def get_exif(dng_file: str):
        with open(dng_file, 'rb') as f:
            tags = exifread.process_file(f, details=False)
            make = str(tags.get('Image Make', '')).lower()
            model = str(tags.get('Image Model', '')).lower()
            f_number = float(str(tags.get('EXIF FNumber', '0')).split('/')[0])
            focal_length = float(str(tags.get('EXIF FocalLength', '0')).split('/')[0])
            return {'make': make, 'model': model, 'f_number': f_number, 'focal_length': focal_length}
    exif_metadata = get_exif(file)
    with rawpy.imread(file) as raw:
        return {'black_level': raw.black_level_per_channel, 'white_level': float(raw.white_level), 'color_matrix': process_color_matrix(raw.color_matrix), 'pattern': get_pattern(raw.color_desc.decode('utf-8'), raw.raw_pattern), 'illum_color': wb_gains_to_illum_color(raw.camera_whitebalance), 'daylight_illum_color': wb_gains_to_illum_color(raw.daylight_whitebalance), 'color_desc': raw.color_desc.decode('utf-8'), 'raw_pattern': raw.raw_pattern, 'camera_make': exif_metadata['make'], 'model': exif_metadata['model'], 'f_number': exif_metadata['f_number'], 'focal_length': exif_metadata['focal_length']}

def extract_image_from_dng(file: str) -> np.ndarray:
    raw = rawpy.imread(file)
    return raw.raw_image_visible.astype(np.float32)

def imwrite(image: np.ndarray, output_path: str, format: str) -> str:
    format = format.lower()
    image = clip(image)
    ext_map = {'png-16': ('.png', np.uint16, 16), 'png-8': ('.png', np.uint8, 8), 'jpeg': ('.jpg', np.uint8, 8), 'jpg': ('.jpg', np.uint8, 8)}
    if format in ext_map:
        ext, dtype, bit_depth = ext_map[format]
        image = (image * (2 ** bit_depth - 1)).astype(dtype)
    else:
        raise ValueError(f"Format '{format}' is not supported.")
    output_file = f'{os.path.splitext(output_path)[0]}{ext}'
    cv2.imwrite(output_file, convert_bgr_rgb(image))
    return output_file

def min_max_normalization(data: np.ndarray, min_vector: np.ndarray, max_vector: np.ndarray):
    return ((data - min_vector) / (max_vector - min_vector + EPS)).astype(np.float32)

def convert_np_arrays(data: np.ndarray) -> List:
    if isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, dict):
        return {key: convert_np_arrays(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_np_arrays(item) for item in data]
    return data

def write_json_file(data: Union[Dict, List], file_path: str):
    with open(os.path.splitext(file_path)[0] + '.json', 'w') as f:
        json.dump(convert_np_arrays(data), f, indent=4)

def read_json_file(file_path: str):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def rg_bg_to_illum_rgb(rgbg: Union[np.ndarray, torch.tensor], normalize: Optional[bool]=True) -> Union[np.ndarray, torch.tensor]:

    def normalize_illum(illum: np.ndarray) -> np.ndarray:
        return illum / vect_norm(illum)
    one_tensor = torch.tensor(1.0, dtype=rgbg.dtype, device=rgbg.device).expand(rgbg.size(0))
    illum = torch.stack([rgbg[:, 0], one_tensor, rgbg[:, 1]], dim=1)
    if normalize:
        return normalize_illum(illum)
    else:
        return illum

def compute_edges(img: np.ndarray) -> np.ndarray:
    edge_img = np.zeros(img.shape)
    img_pad = cv2.copyMakeBorder(img, top=1, bottom=1, left=1, right=1, borderType=cv2.BORDER_REFLECT)
    offsets = [-1, 0, 1]
    for dx, dy in itertools.product(offsets, repeat=2):
        if dx == 0 and dy == 0:
            continue
        edge_img[:, :, :] += np.abs(img[:, :, :] - img_pad[1 + dx:img.shape[0] + 1 + dx, 1 + dy:img.shape[1] + 1 + dy, :])
    edge_img /= 8
    return edge_img

def im2double(img: np.ndarray) -> np.ndarray:
    if img[0].dtype == 'uint8':
        max_value = 255
    elif img[0].dtype == 'uint16':
        max_value = 65535
    else:
        raise ValueError
    return img.astype('float') / max_value

def angular_loss(predicted, gt, shrink=True):
    if isinstance(predicted, list):
        predicted = torch.tensor(predicted).unsqueeze(0)
    if isinstance(gt, list):
        gt = torch.tensor(gt).unsqueeze(0)
    cossim = torch.clamp(torch.sum(predicted * gt, dim=1) / (torch.norm(predicted, dim=1) * torch.norm(gt, dim=1) + EPS), -1, max=1.0)
    if shrink:
        angle = torch.acos(cossim * SHRINK_FACTOR)
    else:
        angle = torch.acos(cossim)
    angular_error = 180.0 / PI * angle
    angular_error = torch.sum(angular_error) / angular_error.shape[0]
    return angular_error

def imread(img_file):
    img = cv2.imread(img_file, -1)
    img = convert_bgr_rgb(img)
    img = im2double(img)
    return img

def imresize(img: np.ndarray, height: int, width: int, interpolation_method: Optional[str]='linear') -> np.ndarray:
    if img.shape[:2] == (height, width):
        return img
    interpolation_methods = {'linear': cv2.INTER_LINEAR, 'bicubic': cv2.INTER_CUBIC, 'nearest': cv2.INTER_NEAREST}
    interpolation = interpolation_methods.get(interpolation_method.lower(), cv2.INTER_LINEAR)
    return cv2.resize(img, (width, height), interpolation=interpolation)

def to_tensor(img):
    dims = len(img.shape)
    assert dims == 3 or dims == 4
    if dims == 3:
        img = img.transpose((2, 0, 1))
    elif dims == 4:
        img = img.transpose((3, 2, 0, 1))
    if dims > 3:
        raise NotImplementedError
    return torch.from_numpy(img)

def vect_norm(vec: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    is_tensor = torch.is_tensor(vec)
    return torch.linalg.vector_norm(vec, dim=1, keepdim=True) if is_tensor else np.linalg.norm(vec)

def convert_bgr_rgb(img: np.ndarray) -> np.ndarray:
    return img[..., ::-1]

def print_args_table(args):
    args_dict = vars(args)
    table_data = [(key, value) for key, value in args_dict.items()]
    headers = ['Argument', 'Value']
    print(tabulate(table_data, headers=headers, tablefmt='grid'))

def compute_snr(img: np.ndarray, patch_size: Optional[int]=15, stride: Optional[int]=1) -> np.ndarray:
    img_tensor = torch.from_numpy(img).float()
    if img_tensor.ndimension() != 3:
        raise ValueError('Input image must be a 3D array with shape (H, W, C).')
    img_tensor = img_tensor.permute(2, 0, 1)
    pad_height = (patch_size - stride) // 2
    pad_width = (patch_size - stride) // 2
    img_padded = torch.nn.functional.pad(img_tensor, (pad_width, pad_width, pad_height, pad_height), mode='reflect')
    patches = img_padded.unfold(1, patch_size, stride).unfold(2, patch_size, stride)
    patches = patches.contiguous().view(img_tensor.size(0), -1, patch_size * patch_size)
    signal_mean = patches.mean(dim=-1, keepdim=True)
    noise_std = patches.std(dim=-1, keepdim=True)
    snr_map = 10 * torch.log10(signal_mean / (noise_std + EPS))
    num_patches_height = (img_tensor.size(1) + 2 * pad_height - patch_size) // stride + 1
    num_patches_width = (img_tensor.size(2) + 2 * pad_width - patch_size) // stride + 1
    snr_map = snr_map.view(img_tensor.size(0), num_patches_height, num_patches_width)
    snr_map = snr_map.permute(1, 2, 0)
    snr_map = torch.clamp(snr_map / 100, 0, 1)
    return snr_map.numpy()

def raw_to_lsrgb(img: Union[np.ndarray, torch.Tensor], illum_color: Union[np.ndarray, List, torch.Tensor], ccm: Union[np.ndarray, List, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    if isinstance(img, np.ndarray) and (isinstance(illum_color, np.ndarray) or isinstance(illum_color, list)) and (isinstance(ccm, np.ndarray) or isinstance(ccm, list)):
        if isinstance(illum_color, list):
            illum_color = np.array(illum_color, dtype=img.dtype)
        if isinstance(ccm, list):
            ccm = np.array(ccm, dtype=img.dtype)
        img_reshaped = img.reshape(-1, 3)
        wb_gain = illum_color[1] / illum_color
        img_wb = img_reshaped @ np.diag(wb_gain)
        img_ccm = img_wb @ ccm.T
        img_out = img_ccm.reshape(img.shape)
        return np.clip(img_out, 0.0, 1.0)
    elif isinstance(img, torch.Tensor) and isinstance(illum_color, torch.Tensor) and isinstance(ccm, torch.Tensor):
        b, c, h, w = img.shape
        wb_gains = illum_color[:, 1].unsqueeze(-1) / (illum_color + EPS)
        img_wb = img * wb_gains.view(b, 3, 1, 1)
        img_wb = img_wb.permute(0, 2, 3, 1).reshape(b, -1, 3)
        img_ccm = torch.bmm(img_wb, ccm.transpose(1, 2))
        img_out = img_ccm.view(b, h, w, 3).permute(0, 3, 1, 2)
        return torch.clamp(img_out, 0.0, 1.0)
    else:
        raise TypeError('Input must be a NumPy array or Torch tensor.')

def compute_2d_rgbg_histogram(img: np.ndarray, hist_boundaries: List[float], hist_bins: int, edge_hist: bool, uv_coord: bool) -> np.ndarray:
    img_reshaped = img.reshape([-1, 3])
    u_min, v_min, u_max, v_max = hist_boundaries
    brightness = np.linalg.norm(img_reshaped, axis=1)
    img_chroma = rgb_to_rgbg(img_reshaped)
    hist, _, _ = np.histogram2d(img_chroma[:, 0], img_chroma[:, 1], bins=hist_bins, range=((u_min, u_max), (v_min, v_max)), weights=brightness)
    norm_factor = np.sum(hist) + EPS
    hist = np.sqrt(hist / norm_factor)
    if uv_coord:
        x_bin_indices, y_bin_indices = np.meshgrid(np.arange(hist_bins), np.arange(hist_bins), indexing='ij')
        x_bin_indices = x_bin_indices / hist_bins
        y_bin_indices = y_bin_indices / hist_bins
        hist = np.stack([hist, x_bin_indices, y_bin_indices], axis=-1)
    if edge_hist:
        edge_img = compute_edges(img)
        edge_chroma = rgb_to_rgbg(edge_img).reshape([-1, 2])
        edge_hist, _, _ = np.histogram2d(edge_chroma[:, 0], edge_chroma[:, 1], bins=hist_bins, range=((u_min, u_max), (v_min, v_max)), weights=brightness)
        norm_factor = np.sum(edge_hist) + EPS
        edge_hist = np.sqrt(edge_hist / norm_factor)
        if uv_coord:
            hist = np.concatenate([hist, np.expand_dims(edge_hist, axis=-1)], axis=-1)
        else:
            hist = np.stack([hist, edge_hist], axis=-1)
    return hist

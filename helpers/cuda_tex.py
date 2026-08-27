import cupy as cp
from cupy.cuda import texture, runtime

_ADDR_MAP = {
    0: runtime.cudaAddressModeWrap,
    1: runtime.cudaAddressModeClamp,
    2: runtime.cudaAddressModeMirror,
    3: runtime.cudaAddressModeBorder
}

_FILTER_MAP = {
    0: runtime.cudaFilterModePoint,
    1: runtime.cudaFilterModeLinear
}


def create_texture_array_2d(img_cp, num_of_chs, textureDescriptorconfig, is_half=False):
    if num_of_chs == 3:
        raise ValueError('CUDA Texture does not support 3-channel-textures.')
    if not isinstance(textureDescriptorconfig, (tuple, list)) or len(textureDescriptorconfig) != 4:
        raise ValueError('Invalid config input.')
    addr_x_raw, addr_y_raw, filter_raw, norm = textureDescriptorconfig
    if addr_x_raw not in _ADDR_MAP:
        raise ValueError(f"Invalid horizontal addressing mode (addr_x): '{addr_x_raw}'。\nValid values: {list(_ADDR_MAP.keys())}")
    if addr_y_raw not in _ADDR_MAP:
        raise ValueError(f"Invalid vertical addressing mode (addr_y): '{addr_y_raw}'。\nValid values: {list(_ADDR_MAP.keys())}")
    if filter_raw not in _FILTER_MAP:
        raise ValueError(f"Invalid filter mode (filter_mode): '{filter_raw}'。\nValid values: {list(_FILTER_MAP.keys())}")
    addr_x = _ADDR_MAP[addr_x_raw]
    addr_y = _ADDR_MAP[addr_y_raw]
    filter_mode = _FILTER_MAP[filter_raw]
    h, w, c = img_cp.shape
    data_contiguous = cp.ascontiguousarray(img_cp, dtype=cp.float16) if is_half else cp.ascontiguousarray(img_cp, dtype=cp.float32)
    bits_per_ch = 16 if is_half else 32
    tup_chn = (bits_per_ch,) * num_of_chs + (0,) * (4 - num_of_chs)
    ch_fmt = texture.ChannelFormatDescriptor(*tup_chn, runtime.cudaChannelFormatKindFloat)
    cuda_arr = texture.CUDAarray(ch_fmt, w, h)
    data_for_copy = data_contiguous.reshape(h, w * c)
    cuda_arr.copy_from(data_for_copy)
    res_desc = texture.ResourceDescriptor(
        runtime.cudaResourceTypeArray, cuArr=cuda_arr
    )
    tex_desc = texture.TextureDescriptor(
        addressModes=(addr_x, addr_y),
        filterMode=filter_mode,
        readMode=runtime.cudaReadModeElementType,
        normalizedCoords=norm,
    )
    return texture.TextureObject(res_desc, tex_desc)


def create_texture_array_3d(img_cp, num_of_chs, textureDescriptorconfig, is_half=False):
    if num_of_chs == 3:
        raise ValueError('CUDA Texture does not support 3-channel-textures.')
    if not isinstance(textureDescriptorconfig, (tuple, list)) or len(textureDescriptorconfig) != 5:
        raise ValueError('Invalid config input.')
    addr_1_raw, addr_2_raw, addr_3_raw, filter_raw, norm = textureDescriptorconfig
    if addr_1_raw not in _ADDR_MAP:
        raise ValueError(f"Invalid dim1 addressing mode (addr_1): '{addr_1_raw}'。\nValid values: {list(_ADDR_MAP.keys())}")
    if addr_2_raw not in _ADDR_MAP:
        raise ValueError(f"Invalid dim2 addressing mode (addr_2): '{addr_2_raw}'。\nValid values: {list(_ADDR_MAP.keys())}")
    if addr_3_raw not in _ADDR_MAP:
        raise ValueError(f"Invalid dim3 addressing mode (addr_3): '{addr_3_raw}'。\nValid values: {list(_ADDR_MAP.keys())}")
    if filter_raw not in _FILTER_MAP:
        raise ValueError(f"Invalid filter mode (filter_mode): '{filter_raw}'。\nValid values: {list(_FILTER_MAP.keys())}")
    addr_x = _ADDR_MAP[addr_1_raw]
    addr_y = _ADDR_MAP[addr_2_raw]
    addr_z = _ADDR_MAP[addr_3_raw]
    filter_mode = _FILTER_MAP[filter_raw]
    h, w, d, c = img_cp.shape
    data_contiguous = cp.ascontiguousarray(img_cp, dtype=cp.float16) if is_half else cp.ascontiguousarray(img_cp, dtype=cp.float32)
    bits_per_ch = 16 if is_half else 32
    tup_chn = (bits_per_ch,) * num_of_chs + (0,) * (4 - num_of_chs)
    ch_fmt = texture.ChannelFormatDescriptor(*tup_chn, runtime.cudaChannelFormatKindFloat)
    cuda_arr = texture.CUDAarray(ch_fmt, d, w, h)
    data_for_copy = data_contiguous.reshape(h, w, d * c)
    cuda_arr.copy_from(data_for_copy)
    res_desc = texture.ResourceDescriptor(
        runtime.cudaResourceTypeArray, cuArr=cuda_arr
    )
    tex_desc = texture.TextureDescriptor(
        addressModes=(addr_x, addr_y, addr_z),
        filterMode=filter_mode,
        readMode=runtime.cudaReadModeElementType,
        normalizedCoords=norm,
    )
    return texture.TextureObject(res_desc, tex_desc)




def create_texture_surface_union_2d(img_cp, num_of_chs, textureDescriptorconfig, is_half=False):
    if num_of_chs == 3:
        raise ValueError('CUDA Texture does not support 3-channel-textures.')
    if not isinstance(textureDescriptorconfig, (tuple, list)) or len(textureDescriptorconfig) != 4:
        raise ValueError('Invalid config input.')
    addr_x_raw, addr_y_raw, filter_raw, norm = textureDescriptorconfig
    if addr_x_raw not in _ADDR_MAP:
        raise ValueError(f"Invalid horizontal addressing mode (addr_x): '{addr_x_raw}'。\nValid values: {list(_ADDR_MAP.keys())}")
    if addr_y_raw not in _ADDR_MAP:
        raise ValueError(f"Invalid vertical addressing mode (addr_y): '{addr_y_raw}'。\nValid values: {list(_ADDR_MAP.keys())}")
    if filter_raw not in _FILTER_MAP:
        raise ValueError(f"Invalid filter mode (filter_mode): '{filter_raw}'。\nValid values: {list(_FILTER_MAP.keys())}")
    addr_x = _ADDR_MAP[addr_x_raw]
    addr_y = _ADDR_MAP[addr_y_raw]
    filter_mode = _FILTER_MAP[filter_raw]
    h, w, c = img_cp.shape
    data_contiguous = cp.ascontiguousarray(img_cp, dtype=cp.float16) if is_half else cp.ascontiguousarray(img_cp, dtype=cp.float32)
    bits_per_ch = 16 if is_half else 32
    tup_chn = (bits_per_ch,) * num_of_chs + (0,) * (4 - num_of_chs)
    ch_fmt = texture.ChannelFormatDescriptor(*tup_chn, runtime.cudaChannelFormatKindFloat)
    cuda_arr = texture.CUDAarray(ch_fmt, w, h,flags=runtime.cudaArraySurfaceLoadStore)
    data_for_copy = data_contiguous.reshape(h, w * c)
    cuda_arr.copy_from(data_for_copy)
    res_desc = texture.ResourceDescriptor(
        runtime.cudaResourceTypeArray, cuArr=cuda_arr
    )
    tex_desc = texture.TextureDescriptor(
        addressModes=(addr_x, addr_y),
        filterMode=filter_mode,
        readMode=runtime.cudaReadModeElementType,
        normalizedCoords=norm,
    )
    return texture.TextureObject(res_desc, tex_desc),texture.SurfaceObject(res_desc)
import h5py, cv2
import numpy as np


def parse_img_array(data):
    """
    Decode an array of byte streams into an image array.

    Args:
        data: np.ndarray of shape (N,), each element is Python bytes or np.ndarray(dtype=uint8)
    Returns:
        imgs: np.ndarray of shape (N, H, W, C), dtype=uint8
    """
    # Ensure data is a 1-D iterable array
    flat = data.ravel()

    imgs = []
    for buf in flat:
        # buf may be bytes or np.ndarray(dtype=uint8)
        if isinstance(buf, (bytes, bytearray)):
            arr = np.frombuffer(buf, dtype=np.uint8)
        elif isinstance(buf, np.ndarray) and buf.dtype == np.uint8:
            arr = buf
        else:
            raise TypeError(f"Unsupported buffer type: {type(buf)}")

        # Decode into a BGR image
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("cv2.imdecode returned None; the byte stream may not be a valid image format")
        imgs.append(img)

    # Stack the list into an ndarray of shape (N, H, W, C)
    return np.stack(imgs, axis=0)


def h5_to_dict(node):
    result = {}
    for name, item in node.items():
        if isinstance(item, h5py.Dataset):
            data = item[()]
            if "rgb" in name:
                result[name] = parse_img_array(data)
            else:
                result[name] = data
        elif isinstance(item, h5py.Group):
            # Recurse into child groups
            result[name] = h5_to_dict(item)
    # If you also want to read attributes, you can:
    if hasattr(node, "attrs") and len(node.attrs) > 0:
        result["_attrs"] = dict(node.attrs)
    return result


def read_hdf5(file_path):
    with h5py.File(file_path, "r") as f:
        data_dict = h5_to_dict(f)
    return data_dict

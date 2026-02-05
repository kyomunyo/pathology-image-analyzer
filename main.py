import numpy as np
from scipy import ndimage as ndi
import matplotlib.pyplot as plt

def otsu_threshold(image):
    image_scaled = (image - image.min()) / (image.max() - image.min()) * 255
    image_scaled = image_scaled.astype(np.uint8)
    hist, bins = np.histogram(image_scaled.ravel(), bins=256, range=(0,256))
    bins_center = (bins[:-1] + bins[1:]) / 2
    total = image.size
    max_var = 0
    thresh = 0
    for i in range(1, 256):
        w1 = np.sum(hist[:i])
        w2 = total - w1
        if w1 == 0 or w2 == 0:
            continue
        m1 = np.sum(hist[:i] * bins_center[:i]) / w1
        m2 = np.sum(hist[i:] * bins_center[i:]) / w2
        var = w1 * w2 * (m1 - m2)**2
        if var > max_var:
            max_var = var
            thresh = bins_center[i]
    thresh_orig = thresh / 255 * (image.max() - image.min()) + image.min()
    return thresh_orig

def calculate_ki67_rate(filename):
    img = plt.imread(filename)
    if img.dtype == np.uint8:
        img = img / 255.0
    img = np.clip(img, 1e-8, 1)
    od = -np.log(img)
    od = np.clip(od, 0, 3)
    H = np.array([0.65002127, 0.70403105, 0.28601261])
    DAB = np.array([0.26799977, 0.5701732 , 0.77642715])
    E = np.array([0.09289866, 0.28188372, 0.95109915])
    M = np.array([H / np.linalg.norm(H),
                  DAB / np.linalg.norm(DAB),
                  E / np.linalg.norm(E)]).T
    M_inv = np.linalg.inv(M)
    od_flat = od.reshape(-1, 3)
    concentrations_flat = M_inv @ od_flat.T
    concentrations_flat = np.maximum(concentrations_flat, 0)
    concentrations = concentrations_flat.T.reshape((od.shape[0], od.shape[1], 3))
    h_channel = concentrations[:,:,0]
    dab_channel = concentrations[:,:,1]
    thresh_h = otsu_threshold(h_channel)
    binary_h = h_channel > thresh_h
    binary_h = ndi.binary_opening(binary_h, np.ones((3,3)))
    binary_h = ndi.binary_closing(binary_h, np.ones((3,3)))
    label_h, num_h = ndi.label(binary_h)
    sizes_h = ndi.sum(binary_h, label_h, range(num_h + 1))
    mask_h = sizes_h > 10
    num_total = np.sum(mask_h)
    if num_h > 0:
        mean_dab = ndi.mean(dab_channel, label_h, range(1, num_h + 1))
        num_positive = np.sum(mean_dab[mask_h[1:]] > 1.5)
    else:
        num_positive = 0
    rate = (num_positive / num_total * 100) if num_total > 0 else 0
    return rate, num_positive, num_total

# 実行例 (filesは添付ファイル名)
files = ['画像_41.jpg', '画像_43.jpg', '画像_44.jpg']
for f in files:
    rate, pos, total = calculate_ki67_rate(f)
    print(f'For {f}: Positive: {pos}, Total: {total}, Rate: {rate:.2f}%')

from Libary import *
from M3_filters import *
# TODO:
""" 
    1. Thêm nhiễu vào ảnh
    2. Xây dựng các phương pháp giảm nhiễu
    3. Cho phép điều chỉnh tham số
    4. Đánh giá ảnh sau xử lý
"""
def add_gaussian_noise(img, mean=0, sigma=5):
    noise = np.random.normal(mean, sigma, img.shape)

    noisy_img = img.astype(np.float64) + noise
    return np.clip(noisy_img, 0, 255).astype(np.uint8)


def add_salt_pepper_noise(img, p, salt_vs_pepper = 0.5):
    noisy_img = img.copy()
    p = p / 100.0
    
    h, w =img.shape[:2]
    
    num_noise = int(p * h *w)
    
    num_salt = int(num_noise * salt_vs_pepper)
    num_pepper = num_noise - num_salt
    
    salt_y = np.random.randint(0, h, num_salt)
    salt_x = np.random.randint(0, w, num_salt)
    
    pepper_y = np.random.randint(0, h, num_pepper)
    pepper_x = np.random.randint(0, w, num_pepper)
    
    noisy_img[salt_y, salt_x] = 255
    noisy_img[pepper_y, pepper_x] = 0
    
    return noisy_img
    
def denoising_phan_dieu_hoa(image, m, n, Q):
    g = image.astype(np.float64) + 1
    ry, rx = m // 2, n // 2
    pad = [(ry, ry), (rx, rx)] + [(0, 0)] * (g.ndim - 2)
    p = np.pad(g, pad, mode="reflect")
    h, w = g.shape[:2]
    out = np.zeros_like(g)

    for y in range(h):
        for x in range(w):
            win = p[y:y + m, x:x + n]          # cửa sổ quanh (x, y)
            num = np.sum(win ** (Q + 1), axis=(0, 1))  # Σ g^(Q+1)
            den = np.sum(win ** Q, axis=(0, 1))        # Σ g^Q
            out[y, x] = num / den - 1
    
    return np.clip(out, 0, 255).astype(np.uint8)
        
def denoising(image, method, ksize=3, q=1.5):

    # if method == "geometric":
    #     # Giữ nguyên phần xử lý trung bình hình học hiện tại
    #     return geometric_mean_filter(image, ksize)

    if method == "contraharmonic":
        return denoising_phan_dieu_hoa(image, ksize, ksize, q)

    else:
        raise ValueError(f"Phương pháp giảm nhiễu không hợp lệ: {method}")


def mse_psnr(image_origin, image_filter):
    mse = np.mean((image_origin.astype(np.float64) - image_filter.astype(np.float64)) ** 2)
    if mse == 0:
        return float("inf")
    psnr = 10 * np.log10(255**2 / mse)
    return mse, psnr
if __name__ == "__main__":
    pass
# 1_fan_crop.py
import os, argparse
import cv2, numpy as np
from PIL import Image
from tqdm import tqdm

def extract_fan_mask(gray):
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    area_ratio = (th>0).sum() / th.size
    if area_ratio < 0.001:
        th = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11,11))
    closed = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)))
    contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        _, th2 = cv2.threshold(gray, 6, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(th2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return np.ones_like(gray, dtype=np.uint8)*255
    c = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(c)
    mask = np.zeros_like(gray, dtype=np.uint8)
    cv2.drawContours(mask, [hull], -1, 255, -1)
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7)), iterations=1)
    return mask

def crop_to_mask(img, mask, pad=8):
    ys, xs = np.where(mask>0)
    if len(xs)==0:
        h,w = img.shape[:2]
        return img, mask, (0,0,w,h)
    x0,x1 = xs.min(), xs.max()
    y0,y1 = ys.min(), ys.max()
    x0 = max(0, x0-pad); y0 = max(0, y0-pad)
    x1 = min(img.shape[1]-1, x1+pad); y1 = min(img.shape[0]-1, y1+pad)
    return img[y0:y1+1, x0:x1+1].copy(), mask[y0:y1+1, x0:x1+1].copy(), (x0,y0,x1-x0+1,y1-y0+1)

def pad_and_resize(crop_img, crop_mask, size):
    h,w = crop_img.shape[:2]
    scale = min(size/w, size/h)
    new_w, new_h = int(w*scale), int(h*scale)
    resized = cv2.resize(crop_img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    resized_mask = cv2.resize(crop_mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    canvas_img = np.zeros((size,size,3), dtype=np.uint8)
    canvas_mask = np.zeros((size,size), dtype=np.uint8)
    x_off = (size - new_w)//2; y_off = (size - new_h)//2
    canvas_img[y_off:y_off+new_h, x_off:x_off+new_w] = resized
    canvas_mask[y_off:y_off+new_h, x_off:x_off+new_w] = resized_mask
    return canvas_img, canvas_mask

def process_folder(src_root, out_root, out_mask_root, size=224, pad=8):
    os.makedirs(out_root, exist_ok=True); os.makedirs(out_mask_root, exist_ok=True)
    for cls in sorted(os.listdir(src_root)):
        in_cls = os.path.join(src_root, cls)
        if not os.path.isdir(in_cls): continue
        out_cls = os.path.join(out_root, cls); os.makedirs(out_cls, exist_ok=True)
        out_mask_cls = os.path.join(out_mask_root, cls); os.makedirs(out_mask_cls, exist_ok=True)
        files = [f for f in os.listdir(in_cls) if f.lower().endswith(('.png','.jpg','.jpeg','.bmp','.tif','.tiff'))]
        for fn in tqdm(files, desc=f"Processing {cls}"):
            src = os.path.join(in_cls, fn)
            img = cv2.imread(src, cv2.IMREAD_COLOR)
            if img is None:
                print("Couldn't read", src); continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            mask = extract_fan_mask(gray)
            masked = cv2.bitwise_and(img, img, mask=mask)
            crop_img, crop_mask, box = crop_to_mask(masked, mask, pad=pad)
            final_img, final_mask = pad_and_resize(crop_img, crop_mask, size)
            cv2.imwrite(os.path.join(out_cls, fn), final_img)
            cv2.imwrite(os.path.join(out_mask_cls, fn), final_mask)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--src", default="LiverDataset")
    p.add_argument("--out", default="LiverDataset_cropped")
    p.add_argument("--out_masks", default="LiverDataset_masks")
    p.add_argument("--size", type=int, default=224)
    p.add_argument("--pad", type=int, default=8)
    args = p.parse_args()
    process_folder(args.src, args.out, args.out_masks, size=args.size, pad=args.pad)
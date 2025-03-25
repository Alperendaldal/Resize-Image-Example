import argparse
import numpy as np
import cv2

def smooth(im):
    height, weight, colorChanel = im.shape
    new_im = np.zeros((height,weight,3),np.uint8)
    filter = np.array([
        [1,1,1],
        [1,1,1],
        [1,1,1]])/9

    pad = 3 // 2
    padded_im = np.pad(im,((pad,pad),(pad,pad),(0,0)), mode='constant')
    for ch in range(colorChanel):
        for i in range(height):
            for j in range(weight):
                region = padded_im[i:i+3,j:j+3,ch]
                new_im[i,j,ch] = np.sum(region * filter)

    return new_im;   

def nn_interpolate(im, c, h, w):
    [px,py] = c

    height, weight, colorChanel = im.shape

    x_cor,y_cor= np.indices((height,weight))
    distances = np.sqrt((x_cor - px) ** 2 + (y_cor - py) ** 2)

    nearest_pixel = np.unravel_index(np.argmin(distances), distances.shape)

    return nearest_pixel


def nn_resize(im, h, w, out_name):
    new_im = np.zeros((h,w,3),dtype=np.uint8)
    height, weight, colorChanel = im.shape
    scaleX = h/height
    scaleY = w/weight

    for X in range(h):
        for Y in range(w):
            c = [X/scaleX,Y/scaleY]
            nearest_pixel = nn_interpolate(im, c,h,w)
            pixel_color = im[nearest_pixel[0], nearest_pixel[1]]
            new_im[X, Y] = pixel_color

    cv2.imwrite(f'{out_name}_nearest_neighbor.png', new_im)
    return new_im


def bilinear_interpolate(im, c, h, w):
    [px,py] = c
    height, weight, colorChanel = im.shape
    X1,X2 = int(np.floor(px)), int(np.ceil(px))
    Y1,Y2 = int(np.floor(py)), int(np.ceil(py))

    X1, X2 = max(0, X1), min(height - 1, X2)
    Y1, Y2 = max(0, Y1), min(weight - 1, Y2)

    closest_4cor = np.array([[X1, Y1], [X1, Y2], [X2, Y1], [X2, Y2]])
                                
    return closest_4cor

def bilinear_resize(im, h, w, out_name):
    height, weight, colorChanel = im.shape
    scaleX = h/height
    scaleY = w/weight
    new_im = np.zeros((h,w,3),dtype=np.uint8)
    for X in range(h):
        for Y in range(w):
            c = [X/scaleX,Y/scaleY]
            pixels = bilinear_interpolate(im,c,h,w)
            P11 = im[pixels[0, 0], pixels[0, 1]]
            P12 = im[pixels[1, 0], pixels[1, 1]]
            P21 = im[pixels[2, 0], pixels[2, 1]]
            P22 = im[pixels[3, 0], pixels[3, 1]]
            
            dx = (c[0] - pixels[0, 0]) / (pixels[1, 0] - pixels[0, 0]) if pixels[1, 0] != pixels[0, 0] else 0
            dy = (c[1] - pixels[0, 1]) / (pixels[2, 1] - pixels[0, 1]) if pixels[2, 1] != pixels[0, 1] else 0

            new_im[X, Y] = np.clip(((1 - dx) * P11 + dx * P21) * (1 - dy) + ((1 - dx) * P12 + dx * P22) * dy, 0, 255).astype(np.uint8)
            
    cv2.imwrite(f'{out_name}_bilinear_resize.png', new_im)
    #write the image in this format: '%s_bilinear' % out_name
    return new_im
    
    
def __main__():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run image resizing.")

    # Required argument for the image filename
    parser.add_argument('img_name', type=str, help="Path to the input image")
    # Required argument for the output filename
    parser.add_argument('out_name', type=str, help="Path to the output image")

    # Optional arguments for resizing dimensions
    parser.add_argument('--width', type=int, default=None, help="Width of the resized image")
    parser.add_argument('--height', type=int, default=None, help="Height of the resized image")

    # Choose between Nearest Neighbor (nn) and Bilinear (bilinear) resizing
    parser.add_argument('--resize_method', type=str, choices=['nn', 'bilinear'], default='nn', help="Resizing method to use")

    args = parser.parse_args()

    # Load the image
    img = smooth(cv2.imread(args.img_name))
    
    if args.width and args.height:
        if args.resize_method == "nn":
            resized_img = nn_resize(img, args.height, args.width, args.out_name)
            print("Resized image using Nearest-Neighbor interpolation.")
        elif args.resize_method == "bilinear":
            resized_img = bilinear_resize(img, args.height, args.width, args.out_name)
            print("Resized image using Bilinear interpolation.")

if __name__ == "__main__":
    __main__()
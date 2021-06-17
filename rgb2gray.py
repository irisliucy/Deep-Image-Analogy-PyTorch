import cv2
import numpy
from skimage import color

base_path = 'data/demo/ppr10k_0/'
filename = '0_2'
img_path = base_path + filename + '.png'
save_to = base_path + 'img_l.png'
correspondence_path = base_path + 'img_AP.png'

def rgb2l(size=(224,224)):
    img = cv2.imread(img_path)
    # img = cv2.imread(img_path)
    corr = cv2.imread(correspondence_path)
    x, y = corr.shape[0], corr.shape[1]
    if img_path.split('.')[-1] == 'tif':
        cv2.imwrite(base_path + filename + '.png', img)
        img = cv2.imread(base_path + filename + '.png')
    
    img_rs_lab = color.rgb2lab(img)
    img_rs_l = img_rs_lab[:,:,[0]]
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized_l = cv2.resize(img_rs_l, size)
    cv2.imwrite(base_path + filename + '_l_resized.png', resized_l)

    # resized = cv2.resize(corr, size)
    # cv2.imwrite(base_path + 'img_AP_resized.png', resized)

def rgb2grayRGB(size=(224,224)):
    name = filename
    print(name)
    img_orig = cv2.imread(img_path)
    x, y = size[0], size[1]
    cv2.imwrite(base_path + name + '.png', img_orig)
    
    # gray = cv2.cvtColor(img_orig, cv2.COLOR_BGR2GRAY)
    # rgb_gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    
    resized_rgb_gray = cv2.resize(img_orig, size, interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(base_path + name + '_gray_resized.png', resized_rgb_gray)

def rgb_resize(size=(224,224)):
    name = filename
    img_orig = cv2.imread(img_path)
    resized_rgb = cv2.resize(img_orig, size, interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(base_path + name + '_rgb_resized.png', resized_rgb)

if __name__ == "__main__":
    rgb_resize()
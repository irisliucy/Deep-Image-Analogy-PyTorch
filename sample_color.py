import cv2
import numpy as np
from skimage import color
import torch
import imutils
import matplotlib.pyplot as plt
from scipy.ndimage.interpolation import zoom


base_path = 'data/demo/ppr10k_0/'
orig_color_img_path = base_path + '0_2_rgb_resized.png'             #'0_1.tif'  # original color image A
orig_target_img_path = base_path + '0_1_rgb_resized.png'           # target image input B
img_path = base_path + 'img_AP_resized.png'                         # image to sample color / deep analogy output  
correspondence_path = base_path + 'img_AP.png'                      # image to sample color
dense_correspondence_npy_path = base_path + 'im_dense_corr_AB.npy'
gray_path = base_path + 'img_l.png'                                 # gray image in rgb
gray_resized_path = base_path + 'img_l_resized.png'
colorized_output_path = base_path + 'colorized_output.png'

sample_mode = ['dense', 'sparse', 'block'][2]         
sample_size = 8
block_size = 3

def lab2rgb_clip(in_lab):
	# return np.clip(color.lab2rgb(in_lab),0,1)
    return np.uint8(np.clip(color.lab2rgb(in_lab),0,1)*255)

def save_resize_img(img_path, img_size, save_as):
    resized_img = cv2.resize(img_path, img_size, interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(base_path + save_as, resized_img)

def rgb2gray(img_path, img_size=(224,224), correspondence_path=None, base_path=base_path):
    if correspondence_path != None:  # provide correspondence if the original image is not png/ jpg
        img_orig = cv2.imread(img_path)
        corr = cv2.imread(correspondence_path)
        x, y = corr.shape[0], corr.shape[1]
        cv2.imwrite(base_path + 'rgb.png', img_orig)
        img = cv2.imread(base_path + 'rgb.png')
    else:
        img = cv2.imread(img_path)

    # saving full resolution grayscale image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # img_rs_lab = color.rgb2lab(img)
    # img_rs_l = img_rs_lab[:,:,[0]]
    cv2.imwrite(base_path + 'gray.png', gray)
    save_resize_img(gray, img_size, save_as='gray_resized.png')

    if correspondence_path != None:
        save_resize_img(corr, img_size, save_as='img_AP_resized.png')

def euclidean_dist(pt1, pt2):
    return np.linalg.norm(pt1 - pt2)

def compute_similarity_matrix(correspondence_np_path):
    """
    correspondence_np_path (numpy): Numpy of the correspondence matrx between the original (A) image and the deep analogy output (AB)
    """
    correspondence_A_AB = np.load(correspondence_np_path)
    x, y = correspondence_A_AB.shape[0], correspondence_A_AB.shape[1]
    print('x: {}, y: {}'.format(x, y))

    distance_mtrx = np.zeros((x, y))
    # corr_A = np.zeros((x, y))
    # corr_AB = np.zeros((x, y, 2))

    for i in range(x):
        for j in range(y):
            corr_A = np.array((i, j))
            # print(corr_A)
            corr_AB = correspondence_A_AB[i,j,:]
            # print(corr_AB)
            distance_mtrx[i,j] = euclidean_dist(corr_A, corr_AB)  # scalar distance calculation
    
    np.save(base_path + 'similarity_mtrx.npy', distance_mtrx)
    print(distance_mtrx.shape)
    print(distance_mtrx.mean())
    print(distance_mtrx.max())

    return distance_mtrx

def compute_off_regions(similarity_mapping):
    # define wrong regions as distance > thres
    distance_thres = similarity_mapping.mean() 
    print('distance threshold : {}'.format(distance_thres))
    off_regions = np.argwhere(similarity_mapping > distance_thres)
    print('Number of off-regions: {}'.format(len(off_regions)))
    return off_regions

def visualize_wrong_regions(img_path, wrong_regions):
    img_rgb = cv2.imread(img_path)[:,:,::-1]
    # img_rgb = cv2.resize(img_rgb, (img_rgb.shape[0]//2, img_rgb.shape[1]//2), interpolation=cv2.INTER_CUBIC)
    
    # create empty image arrays
    print('resized image shape: ', img_rgb.shape)
    x, y = img_rgb.shape[0], img_rgb.shape[1]
    im_ab_ = np.zeros((x, y, 2))
    im_ab = np.zeros((x, y, 2))

    lab_image = color.rgb2lab(img_rgb)
    img_rs_l = lab_image[:,:,[0]]
    im_ab_[:,:,[0]] = lab_image[:,:,[1]]
    im_ab_[:,:,[1]] = lab_image[:,:,[2]]
    img_result = img_rgb.copy()

    for cntr in wrong_regions:
        x , y = cntr[0], cntr[1]
        im_ab[x, y, :] = im_ab_[x,y,:] #[255,0,0]
    
    in_ab_lab_img = np.concatenate((img_rs_l, im_ab), axis=2)
    in_ab_rgb_img = lab2rgb_clip(in_ab_lab_img)

    img_source_input = cv2.imread(orig_color_img_path)[:,:,::-1]
    img_target_input = cv2.imread(orig_target_img_path)[:,:,::-1]

    fig = plt.figure(figsize=(18,6))
    plt.subplot(1,4,1)
    plt.imshow(img_source_input)
    plt.title('Source image input')
    plt.axis('off')

    plt.subplot(1,4,2)
    plt.imshow(img_target_input)
    plt.title('Target image input')
    plt.axis('off')

    plt.subplot(1,4,3)
    plt.imshow(img_rgb)
    plt.title('Deep Analogy output')
    plt.axis('off')

    plt.subplot(1,4,4)
    plt.imshow(in_ab_rgb_img)
    plt.title('Visualization of off-region mappings')
    plt.axis('off')

    fig.tight_layout()

    plt.savefig(base_path + 'off_regions_visualization.png')

def img2fullres(img_path, orig_img_path):
    img = cv2.imread(img_path)
    orig_img = cv2.imread(orig_img_path)

    H_orig, W_orig = orig_img.shape[0], orig_img.shape[1]
    H_proc, W_proc = img.shape[0], img.shape[1]

    # resize output to original resolution
    out_orig_ab = zoom(img, (1.*H_orig/H_proc, 1.*W_orig/W_proc, 1))
    cv2.imwrite(base_path + 'colorized_output_fullres.png', out_orig_ab)
    return out_orig_ab

def bilinear_interpolate_numpy(im, x, y):
    x0 = np.floor(x).astype(int)
    x1 = x0 + 1
    y0 = np.floor(y).astype(int)
    y1 = y0 + 1

    x0 = np.clip(x0, 0, im.shape[1]-1)
    x1 = np.clip(x1, 0, im.shape[1]-1)
    y0 = np.clip(y0, 0, im.shape[0]-1)
    y1 = np.clip(y1, 0, im.shape[0]-1)

    Ia = im[ y0, x0 ]
    Ib = im[ y1, x0 ]
    Ic = im[ y0, x1 ]
    Id = im[ y1, x1 ]

    wa = (x1-x) * (y1-y)
    wb = (x1-x) * (y-y0)
    wc = (x-x0) * (y1-y)
    wd = (x-x0) * (y-y0)

    return (Ia.T*wa).T + (Ib.T*wb).T + (Ic.T*wc).T + (Id.T*wd).T

def find_valid_blocks(coords, block_size):
    '''Given the block size and arrays of valid coordinates, return arrays of valid blocks

    '''
    raise NotImplementedError

if __name__ == "__main__":
    print('[Info] computing similarity matrx...')
    similarity_mtrx = compute_similarity_matrix(dense_correspondence_npy_path)
    print('Size of similarity matrix: {}'.format(similarity_mtrx.shape))
    print('[Info] computing and saving off-regions visualization...')
    off_regions = compute_off_regions(similarity_mtrx)
    visualize_wrong_regions(correspondence_path, off_regions)

    # print('saving original image to grayscale image...')
    # rgb2gray(orig_color_img_path, correspondence_path=correspondence_path, base_path=base_path)

    # resize image
    # im = cv2.imread(base_path + '1019_0.tif')
    # save_resize_img(im, (224,224), save_as='1019_0_resized.png')

    print('[Info] start sampling colors...')
    im_rgb = cv2.imread(img_path)[:,:,::-1]
    x, y = im_rgb.shape[0], im_rgb.shape[1]
    print('x: {}, y: {}'.format(x, y))

    # initialize blank ab input, mask
    im_ab = np.zeros((x, y, 2))
    im_ab_ = np.zeros((x, y, 2))
    im_mask = np.zeros((x, y, 1))

    # generate ab channel mask
    lab_image = color.rgb2lab(im_rgb)
    img_rs_l = lab_image[:,:,[0]]
    im_ab_[:,:,[0]] = lab_image[:,:,[1]]
    im_ab_[:,:,[1]] = lab_image[:,:,[2]]
    
    X,Y = np.where(similarity_mtrx <= similarity_mtrx.mean())   # samples are drawn not from the wrong regions
    ratio_x, ratio_y = x / similarity_mtrx.shape[0], y / similarity_mtrx.shape[1]
    # convert coordinate from dense correspondece to current image # TODO
    assert similarity_mtrx.shape[0] == x
    assert similarity_mtrx.shape[1] == y
    if ratio_x == 1 and ratio_y == 1:
        coords = np.column_stack((X, Y))
    else:
        coords = np.column_stack((X * ratio_x, Y * ratio_y))
    np.random.shuffle(coords)

    if sample_mode == 'sparse':          # random sampling 
        for i in range(sample_size):
            x, y = list(coords)[i]
            x1, y1 = bilinear_interpolate_numpy(im_ab_, x, y)   # result gives the color directly at (x, y)

            # im_ab[x,y,:] = [x1, y1]      # ab channel mask 
            im_ab[x,y,:] = im_ab_[x,y,:]   # ab channel mask
            im_mask[x,y,:] = 1             # binary mask 
    elif sample_mode == 'block':
        for i in range(sample_size):
            x, y = list(coords)[i]
            print('selected px ', im_ab_[x,y,:])
            # average vote
            avg_color_i = np.mean(im_ab_[x:x+block_size,y:y+block_size,[0]])
            avg_color_j = np.mean(im_ab_[x:x+block_size,y:y+block_size,[1]])
            avg_color = [avg_color_i, avg_color_j]
            print('avg color: ', avg_color)
            im_ab[x:x+block_size,y:y+block_size,:] = avg_color
	        # in_ab[x:x+block_size,y:y+block_size,:] = ab[1]
            im_mask[x:x+block_size,y:y+block_size,:] = 1
            

    elif sample_mode == 'dense':
        im_ab[...] = im_ab_                # ab channel mask
        im_mask[...] = 1                   # binary mask 
        
    # check gray + hint color 
    in_ab_rgb_flat = np.concatenate((im_mask*50, im_ab), axis=2)
    in_ab_rgb_flat = lab2rgb_clip(in_ab_rgb_flat)
    print('gray_hint shape: ', in_ab_rgb_flat.shape)
    cv2.imwrite(base_path + 'input_hint.png', in_ab_rgb_flat[:,:,::-1])

    in_ab_lab_img = np.concatenate((img_rs_l, im_ab), axis=2)
    in_ab_rgb_img = lab2rgb_clip(in_ab_lab_img)
    print('gray_hint shape: ', in_ab_rgb_img.shape)
    cv2.imwrite(base_path + 'input_grayscale_hint.png', in_ab_rgb_img[:,:,::-1])
   
    # im_ab = np.transpose(im_ab, (2,0,1))
    print(im_ab.shape)
    np.save(base_path + 'im_ab.npy', im_ab)

    # im_mask = np.transpose(im_mask, (2,0,1))
    print(im_mask.shape) 
   
    np.save(base_path + 'im_mask.npy', im_mask)

    # restore image to full resolution
    # img2fullres('data/demo/ppr10k_1019/colorized_output.png', 'data/demo/ppr10k_1019/1019_0.tif')
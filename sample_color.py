import cv2
import numpy as np
from skimage import color
import torch
import imutils
import matplotlib.pyplot as plt
from scipy.ndimage.interpolation import zoom


base_path = 'data/demo/ppr10k_1019/'
orig_color_img_path = base_path + '1019_0.tif' #'0_1.tif'  # original color image
img_path = base_path + 'img_AP_resized.png'  # image to sample color / deep analogy output  
correspondence_path = base_path + 'img_AP.png'  # image to sample color
gray_path = base_path + 'gray.png'           # gray image in rgb
gray_resized_path = base_path + 'gray_resized.png'
colorized_output_path = base_path + 'colorized_output.png'
sample_mode = ['dense', 'sparse'][1]         
sample_size = 500

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
    print(distance_mtrx)
    print(distance_mtrx.mean())
    print(distance_mtrx.max())

    return distance_mtrx

def compute_off_regions(similarity_mapping):
    # define wrong regions as distance > thres
    distance_thres = similarity_mapping.mean() 
    off_regions = np.argwhere(similarity_mapping > distance_thres)
    print('NUmber of off-regions: {}'.format(len(off_regions)))
    return off_regions

def visualize_wrong_regions(img_path, wrong_regions):
    img_rgb = cv2.imread(img_path)[:,:,::-1]
    img_result = img_rgb.copy()

    for cntr in wrong_regions:
        x , y = cntr[0], cntr[1]
        w , h = 0, 0
        cv2.rectangle(img_result, (x, y), (x+w, y+h), (255, 0, 0), 2)  # bounding box in red
        # print("x,y,w,h:",x,y,w,h)
    
    # Add opacity to the color
    img_result_opac = img_result.copy()
    alpha = 0.9
    mask = img_result.astype(bool)
    img_result_opac[mask] = cv2.addWeighted(img_rgb, alpha, img_result, 1 - alpha, 0)[mask]


    fig = plt.figure(figsize=(18,6))
    plt.subplot(1,2,1)
    plt.imshow(img_rgb)
    plt.title('Colorized output')
    plt.axis('off')

    plt.subplot(1,2,2)
    plt.imshow(img_result_opac)
    plt.title('Visualization of off-region mappings')
    plt.axis('off')

    fig.tight_layout()

    plt.savefig('off_regions_visualization.png')

def img2fullres(img_path, orig_img_path):
    img = cv2.imread(img_path)
    orig_img = cv2.imread(orig_img_path)

    H_orig, W_orig = orig_img.shape[0], orig_img.shape[1]
    H_proc, W_proc = img.shape[0], img.shape[1]

    # resize output to original resolution
    out_orig_ab = zoom(img, (1.*H_orig/H_proc, 1.*W_orig/W_proc, 1))
    cv2.imwrite(base_path + 'colorized_output_fullres.png', out_orig_ab)
    return out_orig_ab

if __name__ == "__main__":
    # print('computing similarity matrx...')
    # similarity_mtrx = compute_similarity_matrix('im_dense_corr_AB.npy')
    
    # # print('computing and saving off-regions visualization...')
    # # off_regions = compute_off_regions(similarity_mtrx)
    # # visualize_wrong_regions(colorized_output_path, off_regions)

    # print('saving original image to grayscale image...')
    # rgb2gray(orig_color_img_path, correspondence_path=correspondence_path, base_path=base_path)

    # resize image
    im = cv2.imread(base_path + '1019_0.tif')
    save_resize_img(im, (224,224), save_as='1019_0_resized.png')

    # print('start sampling colors...')
    # im_rgb = cv2.imread(img_path)[:,:,::-1]
    # x, y = im_rgb.shape[0], im_rgb.shape[1]
    # print('x: {}, y: {}'.format(x, y))

    # # initialize blank ab input, mask
    # im_ab = np.zeros((x, y, 2))
    # im_ab_ = np.zeros((x, y, 2))
    # im_mask = np.zeros((x, y, 1))

    # # generate ab channel mask
    # lab_image = color.rgb2lab(im_rgb)
    # im_ab_[:,:,[0]] = lab_image[:,:,[1]]
    # im_ab_[:,:,[1]] = lab_image[:,:,[2]]
   
    # X,Y = np.where(similarity_mtrx <= similarity_mtrx.mean())
    # # X, Y = np.where(im_rgb[...,0]>=0)
    # print(X, Y)
    # coords = np.column_stack((X,Y))
    # np.random.shuffle(coords)

    # if sample_mode == 'sparse':          # random sampling 
    #     # for n, c in enumerate(list(coords)):   # used to check sample regions
    #     #     x, y = c
    #     for i in range(sample_size):
    #         x, y = list(coords)[i]

    #     # for i in range(sample_size):
    #         # x, y = np.random.randint(0,im_rgb.shape[0]-1), np.random.randint(0,im_rgb.shape[1]-1)
            
    #         im_ab[x,y,:] = im_ab_[x,y,:] # ab channel mask
    #         im_mask[x,y,:] = 1           # binary mask 
    # else:
    #     im_ab[...] = im_ab_              # ab channel mask
    #     im_mask[...] = 1                 # binary mask 
        
    # # check gray + hint color 
    # in_ab_lab_img = np.concatenate((im_mask*50, im_ab), axis=2)
    # in_ab_rgb_img = lab2rgb_clip(in_ab_lab_img)
    # print(in_ab_rgb_img.shape)
    # cv2.imwrite(base_path + 'gray_hint.png', in_ab_rgb_img[:,:,::-1])
   
    # # im_ab = np.transpose(im_ab, (2,0,1))
    # print(im_ab.shape)
    # np.save(base_path + 'im_ab.npy', im_ab)

    # # im_mask = np.transpose(im_mask, (2,0,1))
    # print(im_mask.shape) 
   
    # np.save(base_path + 'im_mask.npy', im_mask)

    # img2fullres('data/demo/ppr10k_1019/colorized_output.png', 'data/demo/ppr10k_1019/1019_0.tif')
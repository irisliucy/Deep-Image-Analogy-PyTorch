import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

def ts2np(x):
    x = x.squeeze(0)
    x = x.cpu().numpy()
    x = x.transpose(1,2,0)
    return x

def np2ts(x, device):
    x = x.transpose(2,0,1)
    x = torch.from_numpy(x)
    x = x.unsqueeze(0)
    x = x.to(device)
    return x



def blend(response, f_a, r_bp, alpha=0.8, tau=0.05):
    """
    :param response:
    :param f_a: feature map (either F_A or F_BP)
    :param r_bp: reconstructed feature (R_BP or R_A)
    :param alpha: scalar balance the ratio of content and style in new feature map
    :param tau: threshold, default: 0.05 (suggested in paper)
    :return: (f_a*W + r_bp*(1-W)) where W=alpha*(response>tau)

    Following the official implementation, I replace the sigmoid function (stated in paper) with indicator function
    """
    weight = (response > tau).type(f_a.type()) * alpha
    weight = weight.expand(1, f_a.size(1), weight.size(2), weight.size(3))

    # f_ap = f_a*weight + r_bp*(1. - weight)
    f_ap = f_a*weight
    return f_ap


def normalize(feature_map):
    """

    :param feature_map: either F_a or F_bp
    :return:
    normalized feature map
    response
    """
    response = torch.sum(feature_map*feature_map, dim=1, keepdim=True)
    normed_feature_map = feature_map/torch.sqrt(response)

    # response should be scaled to (0, 1)
    response = (response-torch.min(response))/(torch.max(response)-torch.min(response))
    return  normed_feature_map, response


def load_image(file_A, resizeRatio=1.0):
    ori_AL = cv2.imread(file_A)
    ori_img_sizes = ori_AL.shape[:2]

    # resize
    if (ori_AL.shape[0] > 700):
        ratio = 700 / ori_AL.shape[0]
        ori_AL = cv2.resize(ori_AL,None, fx=ratio, fy=ratio, interpolation=cv2.INTER_CUBIC)

    if (ori_AL.shape[1] > 700):
        ratio = 700 / ori_AL.shape[1]
        ori_AL = cv2.resize(ori_AL,None, fx=ratio, fy=ratio, interpolation=cv2.INTER_CUBIC)

    if (ori_AL.shape[0] < 200):
        ratio = 700 / ori_AL.shape[0]
        ori_AL = cv2.resize(ori_AL,None, fx=ratio, fy=ratio, interpolation=cv2.INTER_CUBIC)
    if (ori_AL.shape[1] < 200):
        ratio = 700 / ori_AL.shape[1]
        ori_AL = cv2.resize(ori_AL,None, fx=ratio, fy=ratio, interpolation=cv2.INTER_CUBIC)

    if (ori_AL.shape[0]*ori_AL.shape[1] > 350000):
        ratio = np.sqrt(350000 / (ori_AL.shape[1]*ori_AL.shape[0]))
        ori_AL = cv2.resize(ori_AL,None, fx=ratio, fy=ratio, interpolation=cv2.INTER_CUBIC)

    img_A = cv2.resize(ori_AL,None, fx=resizeRatio, fy=resizeRatio, interpolation=cv2.INTER_CUBIC)

    return img_A

def output_dense_correspondence(img, orig_img):
    """ Output dense correspondence
    """
    nnf = img

    img = np.zeros((nnf.shape[0], nnf.shape[1], 3), dtype=np.uint8)
    for i in range(nnf.shape[0]):
        for j in range(nnf.shape[1]):
            pos = nnf[i, j]
            img[i, j, 0] = int(255 * (pos[0] / orig_img.shape[1]))
            img[i, j, 2] = int(255 * (pos[1] / orig_img.shape[0]))
    return img

def save_optical_flow_img(img, orig_img, save_to):
    img = output_dense_correspondence(img, orig_img)
    cv2.imwrite(save_to, img)

def warp(x, flo):
    """
    warp an image/tensor (im2) back to im1, according to the optical flow
    # x: [B, C, H, W] (im2)
    # flo: [B, 2, H, W] flow

    x: [C, H, W] (im2)
    flo: [2, H, W] flow
    """
    x = cv2.resize(x, (224, 224))
    x = torch.Tensor(x)
    flo = torch.Tensor(flo)
    
   
    # flo = flo.reshape(1, C, H, W)
    flo = torch.unsqueeze(flo, dim=0)
    B, H, W, C = flo.size()
    flo = flo.permute(0,3,1,2)    
    x = torch.unsqueeze(x, dim=0)
    B, H, W, C = x.size()
    x = x.permute(0,3,1,2)  
    
    # B, C, H, W = x.size()
    # mesh grid 
    xx = torch.arange(0, W).view(1,-1).repeat(H,1)
    yy = torch.arange(0, H).view(-1,1).repeat(1,W)
    xx = xx.view(1,1,H,W).repeat(B,1,1,1)
    yy = yy.view(1,1,H,W).repeat(B,1,1,1)
    grid = torch.cat((xx,yy),1).float()
    # flo = flo.permute(2,0,1)

    if x.is_cuda:
        grid = grid.cuda()
    vgrid = Variable(grid) + flo

    # scale grid to [-1,1] 
    vgrid[:,0,:,:] = 2.0*vgrid[:,0,:,:].clone() / max(W-1,1)-1.0
    vgrid[:,1,:,:] = 2.0*vgrid[:,1,:,:].clone() / max(H-1,1)-1.0
    vgrid = vgrid.permute(0,2,3,1)      

    output = nn.functional.grid_sample(x, vgrid)
    mask = torch.autograd.Variable(torch.ones(x.size())).cuda()
    mask = nn.functional.grid_sample(mask, vgrid)
    # if W==128:
    np.save('im_mask.npy', mask.cpu().data.numpy())
    np.save('im_warp.npy', output.cpu().data.numpy())
    
    mask[mask<0.9999] = 0
    mask[mask>0] = 1
    
    np.save('im_ab.npy', (output*mask).cpu().data.numpy())
    return output*mask

def output_sample_mask(array, orig_img):
    print(array.shape)
    np.save('im_dense_corr.npy', array) 

    # Random Sampling 
    # masked = np.random.randint(0, 1, size=array.shape)
    # np.save('im_mask.npy', masked)

    # Sample from red regions (high cofidence)
    # in this case, select all pixels with a red value > 0.3
    img = output_dense_correspondence(array, orig_img)
    mask = img[..., 0] < 0.3
    # Set all masked pixels to zero
    masked = orig_img.copy()
    masked[mask] = 0
    np.save('im_mask.npy', masked)

    # Mask input image with binary mask
    result = cv2.bitwise_and(orig_img, masked)
    np.save('im_ab.npy', result)
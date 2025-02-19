import cv2
import numpy as np
import torch
import torch.nn.functional as F
from . import flowlib as fl
from models.hd3_ops import *

def boundleIndex(index,bound):
    index=max(0,index)
    index=min(index,bound-1)
    return index
def get_visualization(img_list, label_list, ms_vect, ms_prob, ds=6, idx=0):
    evaluate = True
    if len(label_list)==0:
        evaluate=False
    dim = ms_vect[0].size(1)
    H, W = img_list[0].size()[2:]
    with torch.no_grad():
        raw_img0 = _recover_img(img_list[0][idx].data)
        raw_img1 = _recover_img(img_list[1][idx].data)
        for l in range(len(ms_vect)-1,len(ms_vect)):
            # image
            vis_list = [raw_img0, raw_img1]
            if(evaluate):
                # ground-truth flow
                gt_flo, valid_mask = downsample_flow(label_list[0],1 / 2**(ds - l))
                gt_flo = F.interpolate(gt_flo, (H, W), mode='nearest')[idx]
                valid_mask = F.interpolate(valid_mask, (H, W), mode='nearest')[idx]
                max_mag1 = torch.max(torch.norm(gt_flo, 2, 0))


            # predicted flow
            pred_flo = ms_vect[l]
            if dim == 1:
                pred_flo = disp2flow(pred_flo)
            pred_flo = F.interpolate(pred_flo, (H, W), mode='nearest')[idx]
            max_mag2 = torch.max(torch.norm(pred_flo, 2, 0))
            if (evaluate):
                max_mag = max(float(max_mag1), float(max_mag2))
                pred_flo_np=pred_flo.cpu().numpy()
                gt_flo_np = gt_flo.cpu().numpy()
            else:
                max_mag = float(max_mag2)
                pred_flo_np = pred_flo.cpu().numpy()

            shape = raw_img0.shape
            gt_flo_img =  np.zeros(shape,dtype=np.float32)
            pred_flo_img = np.zeros(shape,dtype=np.float32)
            img1=raw_img1.cpu().numpy()
            for row in range(0, shape[1]):
                for col in range(0, shape[2]):
                    if (evaluate):
                        v = gt_flo_np[:,row, col]
                        gt_flo_img[:,row, col] = img1[:,boundleIndex(int(row + v[1]),shape[1]), boundleIndex(int(col + v[0]),shape[2])]
                    v = pred_flo_np[:,row, col]
                    pred_flo_img[:,row, col] = img1[:,boundleIndex(int(row + v[1]),shape[1]), boundleIndex(int(col + v[0]),shape[2])]
            error=cv2.absdiff(img1,pred_flo_img)
            error=np.transpose(error,[1,2,0])
            if(evaluate):
                gt_flo_img = np.transpose(gt_flo_img, [1, 2, 0])
                r, g, b = cv2.split(gt_flo_img)
                gt_flo_img = cv2.merge([b, g, r])
                cv2.imshow("gt",gt_flo_img)
            pred_flo_img = np.transpose(pred_flo_img, [1,2,0])
            r, g, b = cv2.split(pred_flo_img)
            pred_flo_img=cv2.merge([b,g,r])
            cv2.imshow("err",error)
            cv2.imshow("pred",pred_flo_img)
            cv2.waitKey(100)
            #vis_list.append(error)
            # epe error visualization
            #epe_error = torch.norm(
            #    pred_flo - gt_flo, 2, 0, keepdim=False) * valid_mask[0, :, :]
            #normalizer = max(torch.max(epe_error), 1)
            #epe_error = 1 - epe_error / normalizer
            #vis_list.append(_visualize_heat(epe_error))


            # prob = ms_prob[l].data
            # prob = prob_gather(prob, normalize=True, dim=dim)
            # if prob.size(2) != H or prob.size(3) != W:
            #     prob = F.interpolate(prob, (H, W), mode='nearest')
            # vis_list.append(_visualize_heat(prob[idx].squeeze(), cv2.COLORMAP_BONE))

            #vis = torch.cat(vis_list, dim=2)
            # if l == 0:
            #     ms_vis = vis
            # else:
            #     ms_vis = torch.cat([ms_vis, vis], dim=1)
        return pred_flo_img


def _recover_img(img):
    mean = torch.tensor([0.485, 0.456, 0.406], device=img.device)
    std = torch.tensor([0.229, 0.224, 0.225], device=img.device)
    img = img.permute(1, 2, 0) * std + mean
    return img.permute(2, 0, 1)


def _flow_to_img(flo, mag=-1):
    img = torch.from_numpy(
        fl.flow_to_image(flo.permute(1, 2, 0).cpu().numpy(), mag)).cuda()
    return img.permute(2, 0, 1).float() / 255.0


def _visualize_heat(x, method=cv2.COLORMAP_JET):
    x = np.uint8(x.cpu().numpy() * 255)
    x = torch.from_numpy(cv2.applyColorMap(x, method)).cuda()
    return x.permute(2, 0, 1).float() / 255.0

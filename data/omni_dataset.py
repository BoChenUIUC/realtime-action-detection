# Mathematical
import numpy as np
from scipy.ndimage.interpolation import map_coordinates
import cv2

# Pytorch
import torch
from torch.utils import data
from torchvision import datasets

# Misc
from functools import lru_cache

import os

import sys
sys.path.insert(0, '/home/monet/research/realtime-action-detection')
from utils.augmentations import SSDAugmentation
import collections
from PIL import Image

def genuv(h, w):
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    u = (u + 0.5) * 2 * np.pi / w - np.pi
    v = (v + 0.5) * np.pi / h - np.pi / 2
    return np.stack([u, v], axis=-1)


def uv2xyz(uv):
    sin_u = np.sin(uv[..., 0])
    cos_u = np.cos(uv[..., 0])
    sin_v = np.sin(uv[..., 1])
    cos_v = np.cos(uv[..., 1])
    return np.stack([
        cos_v * cos_u,
        cos_v * sin_u,
        sin_v
    ], axis=-1)


def xyz2uv(xyz):
    c = np.sqrt((xyz[..., :2] ** 2).sum(-1))
    u = np.arctan2(xyz[..., 1], xyz[..., 0])
    v = np.arctan2(xyz[..., 2], c)
    return np.stack([u, v], axis=-1)

def get_rotated_mat(outshape,inshape,rot_x,rot_y,rot_z,fov):
    uv = genuv(*outshape)
    xyz = uv2xyz(uv.astype(np.float64))

    # rotate along x-axis
    xyz_rot = xyz.copy()
    xyz_rot[..., 0] = xyz[..., 0]
    xyz_rot[..., 1] = np.cos(rot_x) * xyz[..., 1] - np.sin(rot_x) * xyz[..., 2]
    xyz_rot[..., 2] = np.sin(rot_x) * xyz[..., 1] + np.cos(rot_x) * xyz[..., 2]
    # rotate along y-axis
    xyz = xyz_rot.copy()
    xyz_rot = xyz.copy()
    xyz_rot[..., 0] = np.cos(rot_y) * xyz[..., 0] - np.sin(rot_y) * xyz[..., 2]
    xyz_rot[..., 1] = xyz[..., 1]
    xyz_rot[..., 2] = np.sin(rot_y) * xyz[..., 0] + np.cos(rot_y) * xyz[..., 2]
    # rotate along z-axis
    xyz = xyz_rot.copy()
    xyz_rot = xyz.copy()
    xyz_rot[..., 0] = np.cos(rot_z) * xyz[..., 0] - np.sin(rot_z) * xyz[..., 1]
    xyz_rot[..., 1] = np.sin(rot_z) * xyz[..., 0] + np.cos(rot_z) * xyz[..., 1]
    xyz_rot[..., 2] = xyz[..., 2]

    # get rotated uv matrix
    uv_rot = xyz2uv(xyz_rot)

    u = uv_rot[..., 0]
    v = uv_rot[..., 1]

    h,w = inshape
    x = np.tan(u)
    y = np.tan(v) / np.cos(u)
    x = x * w / (2 * np.tan(fov / 2)) + w / 2
    y = y * h / (2 * np.tan(fov / 2)) + h / 2

    return u,v,x,y

def get_region(u,v,x,y,fov,xmin,ymin,xmax,ymax):
    invalid = (u < -fov / 2) | (u > fov / 2) |\
              (v < -fov / 2) | (v > fov / 2)
    x[invalid] = -1
    y[invalid] = -1

    valid = (x >= xmin) & (x <= xmax) & (y >= ymin) & (y <= ymax)
    return valid


def IoU(bbox1,bbox2,outshape=(300,600),inshape=(300,300),fov=np.pi/3):
    h,w = inshape
    # get region of the first box
    xmin1,ymin1,xmax1,ymax1,rot_x1,rot_y1,rot_z1 = bbox1
    u1,v1,x1,y1 = get_rotated_mat(outshape,inshape,rot_x1,rot_y1,rot_z1,fov)
    valid1 = get_region(u1,v1,x1,y1,fov,xmin1*w,ymin1*h,xmax1*w,ymax1*h)

    # get region of the second box
    xmin2,ymin2,xmax2,ymax2,rot_x2,rot_y2,rot_z2 = bbox2
    u2,v2,x2,y2 = get_rotated_mat(outshape,inshape,rot_x2,rot_y2,rot_z2,fov)
    valid2 = get_region(u2,v2,x2,y2,fov,xmin2*w,ymin2*h,xmax2*w,ymax2*h)

    intersec = sum(sum(valid1&valid2))
    union = sum(sum(valid1|valid2))
    iou = intersec/union

    return iou



def rotated_coord(uv, h, w, u_fov, v_fov, rot_x=0, rot_y=0, rot_z=0):
    # the coord on sphere of each pixel in the output image
    xyz = uv2xyz(uv.astype(np.float64)) # out_h, out_w, (x,y,z)

    # rotate along z-axis
    xyz_rot = xyz.copy()
    xyz_rot[..., 0] = np.cos(rot_z) * xyz[..., 0] - np.sin(rot_z) * xyz[..., 1]
    xyz_rot[..., 1] = np.sin(rot_z) * xyz[..., 0] + np.cos(rot_z) * xyz[..., 1]
    xyz_rot[..., 2] = xyz[..., 2]
    xyz = xyz_rot.copy()
    # rotate along y-axis
    xyz_rot = xyz.copy()
    xyz_rot[..., 0] = np.cos(rot_y) * xyz[..., 0] - np.sin(rot_y) * xyz[..., 2]
    xyz_rot[..., 1] = xyz[..., 1]
    xyz_rot[..., 2] = np.sin(rot_y) * xyz[..., 0] + np.cos(rot_y) * xyz[..., 2]
    xyz = xyz_rot.copy()
    # rotate along x-axis
    xyz_rot = xyz.copy()
    xyz_rot[..., 0] = xyz[..., 0]
    xyz_rot[..., 1] = np.cos(rot_x) * xyz[..., 1] - np.sin(rot_x) * xyz[..., 2]
    xyz_rot[..., 2] = np.sin(rot_x) * xyz[..., 1] + np.cos(rot_x) * xyz[..., 2]
    # get rotated uv matrix
    uv_rot = xyz2uv(xyz_rot)

    u = uv_rot[..., 0]
    v = uv_rot[..., 1]

    x = np.tan(u)
    y = np.tan(v) / np.cos(u)
    x = x * w / (2 * np.tan(u_fov / 2)) + w / 2
    y = y * h / (2 * np.tan(v_fov / 2)) + h / 2

    invalid = (u < -u_fov / 2) | (u > u_fov / 2) |\
              (v < -v_fov / 2) | (v > v_fov / 2) 
    x[invalid] = -1
    y[invalid] = -1
    # every entry is a coord

    return np.stack([y, x], axis=0),x,y

def GreatCircleDist(uv1, uv2):
    u1,v1 = uv1
    u2,v2 = uv2
    x1,y1,z1 = np.cos(v1)*np.cos(u1),np.cos(v1)*np.sin(u1),np.sin(v1)
    x2,y2,z2 = np.cos(v2)*np.cos(u2),np.cos(v2)*np.sin(u2),np.sin(v2)
    d = ((x1-x2)**2+(y1-y2)**2+(z1-z2)**2)**0.5
    phi = np.arcsin(d/2)
    return phi*2



class OmniDataset(data.Dataset):
    def __init__(self, dataset, fov=120, outshape=(512, 2*512),
                 z_rotate=True, y_rotate=True, x_rotate=False,
                 fix_aug=False, use_background=True, num_bgs=22, save_final_annot=True, use_npy=False):
        '''
        Convert classification dataset to omnidirectional version
        @dataset  dataset with same interface as torch.utils.data.Dataset
                  yield (PIL image, label) if indexing
        '''
        # if use npy file to store images and labels
        if use_npy:
            # save np.load
            np_load_old = np.load

            # modify the default parameters of np.load
            np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)


            img_file,label_file,index_file = 'img_file.npy', 'label_file.npy', 'index_file.npy'
            # with open(img_file, 'rb') as f:
            #     self.imgs = np.load(f)
            self.imgs = None
            with open(label_file, 'rb') as f:
                self.labels = np.load(f)
            with open(index_file, 'rb') as f:
                self.indexes = np.load(f)
            d=np.load("dataset.npy").item()
            self.ids = d['ids']
            self.video_list = d['video_list']
            self.root = d['root']
            self.name = d['name']

            # restore np.load for future normal usage
            np.load = np_load_old
            print(self.indexes.shape[0],"images loaded.")
            return 
        else:
            self.dataset = dataset
            self.ids = dataset.ids
            self.video_list = dataset.video_list
            self.root = dataset.root
            self.name = dataset.name

        self.fov = fov
        self.outshape = outshape
        self.z_rotate = z_rotate
        self.y_rotate = y_rotate
        self.x_rotate = x_rotate
        self.use_background = use_background
        # self.annot_map = collections.defaultdict(dict)
        self.user = '/home/monet/'
        self.use_npy = False


        self.aug = None
        if fix_aug:
            self.aug = [
                {
                    'z_rotate': 0,#np.random.uniform(-np.pi, np.pi),
                    'y_rotate': 0,#np.random.uniform(-np.pi/2, np.pi/2),
                    'x_rotate': 0,#np.random.uniform(-np.pi, np.pi),
                }
                for _ in range(len(self.dataset))
            ]

        # load backgorounds
        self.bg_imgs = []
        img_root = self.user + 'research/realtime-action-detection/data/background/'
        for bg_idx in range(1,23):
            img_name = img_root + str(bg_idx) + '.jpg'
            bg_img = cv2.imread(img_name)
            bg_img = cv2.resize(bg_img, (self.outshape[1], self.outshape[0]))
            self.bg_imgs += [bg_img]

        # map video to background
        L = len(self.video_list)
        self.vid2bgidx = {}
        for vid in range(L):
            self.vid2bgidx[vid] = np.random.randint(0,22)

        # map video to number of frames in a video
        # map video to starting frame index,
        self.video_start = {}
        self.vid2totalframe = collections.defaultdict(int)
        prev_vid = -1
        for i,annot_info in enumerate(self.ids):
            video_id = annot_info[0]
            if video_id != prev_vid:
                self.video_start[video_id] = i
            prev_vid = video_id
            self.vid2totalframe[video_id] += 1

        # map video to another friend video
        self.friend_video = {}
        video_ids = list(self.video_start.keys())
        num_videos = len(video_ids)
        for i in range(num_videos):
            one = video_ids[i]
            two = video_ids[(i + np.random.randint(1,num_videos))%num_videos]
            self.friend_video[one] = two
            assert(one!=two)

        # map video to rotation
        self.vid2rot = {}
        for vid in range(len(self.video_list)):
            if self.y_rotate:
                if self.aug is not None:
                    rot_y = self.aug[vid]['y_rotate']
                else:
                    rot_y = np.random.uniform(-np.pi/2, np.pi/2)
            else:
                rot_y = 0

            if self.z_rotate:
                if self.aug is not None:
                    rot_z = self.aug[vid]['z_rotate']
                else:
                    rot_z = np.random.uniform(-np.pi, np.pi)
            else:
                rot_z = 0

            if self.x_rotate:
                if self.aug is not None:
                    rot_x = self.aug[vid]['x_rotate']
                else:
                    rot_x = np.random.uniform(-np.pi, np.pi)
            else:
                rot_x = 0

            self.vid2rot[vid] = [(rot_x,rot_y,rot_z)]

            while True:
                rot_y2 = np.random.uniform(-np.pi/2, np.pi/2) 
                rot_z2 = np.random.uniform(-np.pi, np.pi)
                rot_x2 = 0
                if GreatCircleDist((rot_z,rot_y),(rot_z2,rot_y2))>=1.7:break
            self.vid2rot[vid] += [(rot_x2,rot_y2,rot_z2)]

        # try to save data in file len(self.ids)
        # print('Start building dataset...')
        # imgs, labels, indexes = [],[],[]
        # prev_vid = -1
        # v_cnt,max_video = 0,50
        # for idx in range(len(self.ids)):
        #     print(idx)
        #     annot_info = self.ids[idx]
        #     video_id = annot_info[0]
        #     if video_id != prev_vid:
        #         prev_vid = video_id
        #         v_cnt += 1
        #         if v_cnt >= max_video:
        #             break
        #     img, label, index = self._transform_item(idx)
        #     pil_img = Image.fromarray(img.cpu().permute(1, 2, 0).numpy().astype(np.uint8))
        #     pil_img.save('../dataset/{:05d}.png'.format(idx))
        #     labels.append(label)
        #     indexes.append(index)

        # labels = np.array(labels)
        # indexes = np.array(indexes)
        # print(indexes.shape)
        # with open('label_file.npy','wb') as f:
        #     np.save(f, labels)
        # with open('index_file.npy','wb') as f:
        #     np.save(f, indexes)
        # d = {}
        # d['ids'] = dataset.ids
        # d['video_list'] = dataset.video_list
        # d['root'] = dataset.root
        # d['name'] = dataset.name
        # np.save("dataset.npy", d)
        # exit(0)

        # end

        # # save jhmdb convereted data in cache
        # if self.dataset.image_set == 'test' and self.name == 'jhmdb':
        #     original_annot_location = self.user + 'research/dataset/ucf24/splitfiles/finalAnnots.mat'
        #     final_annot_location = self.root + 'splitfiles/correctedAnnots_' + self.dataset.image_set + '.mat'
        #     if os.path.exists(final_annot_location):
        #         return
        #     import scipy.io as sio
        #     import copy
        #     old_annots = sio.loadmat(original_annot_location)
        #     annot = old_annots['annot']
        #     template = annot[0][0]
        #     tubes = []
        #     for vid, video in enumerate(self.dataset.vddb):
        #         print(vid,video['video_name'],len(video['gt_bboxes']))
        #         new_tube = copy.deepcopy(template)
        #         new_tube[0][0][0] = len(video['gt_bboxes'])
        #         new_tube[1] = [video['video_name']]
        #         new_tube[2][0][0][0][0][0] = len(video['gt_bboxes'])
        #         new_tube[2][0][0][1][0][0] = 1
        #         new_tube[2][0][0][2][0][0] = video['gt_label'] + 1
        #         new_boxes = []
        #         for fid in range(len(video['gt_bboxes'])):
        #             gt_box = video['gt_bboxes'][fid]
        #             gt_box[0] /= 320
        #             gt_box[1] /= 240
        #             gt_box[2] /= 320
        #             gt_box[3] /= 240
        #             old_label = np.concatenate((gt_box, [video['gt_label']]))

        #             h, w = 300,300
        #             uv = genuv(*self.outshape) # out_h, out_w, (out_phi, out_theta)
        #             fov = self.fov * np.pi / 180

        #             img_idx, x, y = rotated_coord(uv, h, w, fov, fov, *self.vid2rot[vid])

        #             label = self._transform_label([old_label],x, y)[0]
        #             new_boxes.append([int(label[0]*1024),int(label[1]*512),int(label[2]*1024),int(label[3]*512)])
        #         new_tube[2][0][0][3] = new_boxes
        #         tubes.append(new_tube)

        #     sio.savemat(final_annot_location,{'annot':tubes})

        # elif self.dataset.image_set == 'test' and self.name == 'ucf24':
        #     self.original_annot_location = self.root +'splitfiles/finalAnnots.mat'
        #     self.final_annot_location = self.root + 'splitfiles/correctedAnnots_' + self.dataset.image_set + '.mat'
        #     # if os.path.exists(self.final_annot_location):
        #     #     return
        #     print('transforming annotation')
        #     assert(os.path.exists(self.original_annot_location))
        #     import collections
        #     self.annot_map = collections.defaultdict(dict)

        #     # transform the images
        #     for idx in range(len(self.dataset)):
        #         annot_info = self.ids[idx]
        #         video_id = annot_info[0]
        #         videoname = self.video_list[video_id]

        #         img, label, index = self.dataset[idx]

        #         h, w = img.shape[1:]
        #         uv = genuv(*self.outshape) # out_h, out_w, (out_phi, out_theta)
        #         fov = self.fov * np.pi / 180

        #         img_idx, x, y = rotated_coord(uv, h, w, fov, fov, *self.vid2rot[video_id])

        #         label = self._transform_label(label,x, y)
        #         old_label = self.ids[idx][3]
        #         for old,new in zip(old_label,label):
        #             old2 = (int(old[0]),int(old[1]),int(old[2]-old[0]),int(old[3]-old[1]))
        #             if sum(old2) == 0:continue
        #             self.annot_map[videoname][old2] = [int(new[0]*1024),
        #                                                 int(new[1]*512),
        #                                                 int(new[2]*1024),
        #                                                 int(new[3]*512)]
        #         if idx%500 == 0 and idx > 0:
        #             print('Transforming %6d/%6d'%(idx,len(dataset)))
        #             break

        #     # transform the annotation
        #     import scipy.io as sio
        #     old_annots = sio.loadmat(self.original_annot_location)
        #     for annot in old_annots['annot'][0]:
        #         filename = annot[1][0]
        #         if filename in self.annot_map:
        #             for tube in annot[2][0]:
        #                 new_boxes = []
        #                 for i,old_box in enumerate(tube[3]):
        #                     key = (old_box[0],old_box[1],old_box[2],old_box[3])
        #                     if (key in self.annot_map[filename]):
        #                         new_boxes.append(self.annot_map[filename][key])
        #                 tube[3] = new_boxes
        #             print(filename,'yes')
        #         else:
        #             print(filename,'no')
        #     sio.savemat(self.final_annot_location,{'annot':old_annots['annot'][0]})
        #     print('transform finishes')
        #     exit(0)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        if self.use_npy:
            img, label, index = None, torch.from_numpy(self.labels[idx]), self.indexes[idx]
        else:
            img, label, index = self._transform_item(idx)
        return img, label, index

    def get_rotation(self, video_id):
        return self.vid2rot[video_id]

    def _transform_item(self, idx):
        annot_info = self.ids[idx]
        video_id = annot_info[0]
        videoname = self.video_list[video_id]

        # friend video
        fvid = self.friend_video[video_id]
        target_index = self.video_start[fvid] + idx - self.video_start[video_id]
        last_index = self.video_start[fvid] + self.vid2totalframe[fvid] - 1

        # images for patching
        imgs = [self.dataset[idx]]
        if last_index>=target_index:
            imgs += [self.dataset[fvid]]
        else:
            imgs += [None]

        # use background image
        bg_img = None
        if self.use_background:
            bg_idx = self.vid2bgidx[video_id]
            bg_img = self.bg_imgs[bg_idx]

        # prepare for translation
        h = w = 300
        uv = genuv(*self.outshape) # out_h, out_w, (out_phi, out_theta)
        fov = self.fov * np.pi / 180

        invalid = None
        img_stack = [None for ch in range(3)]
        new_labels = []
        for i,(rot_x,rot_y,rot_z) in enumerate(self.vid2rot[video_id]):
            # fetch 2d image
            if imgs[i] is None:break
            img, label, index = imgs[i]
            img_idx, x, y = rotated_coord(uv, h, w, fov, fov, rot_x, rot_y, rot_z)

            if invalid is None:
                invalid = (x < 0) | (x > w) | (y < 0) | (y > h)
            else:
                invalid &= ((x < 0) | (x > w) | (y < 0) | (y > h))

            for ch in range(img.shape[0]):
                tmp = map_coordinates(img[ch,:,:], img_idx, order=1) 
                if img_stack[ch] is None:
                    img_stack[ch] = tmp
                else:
                    img_stack[ch] += tmp

            new_labels += self._transform_label(label,x, y)

        if bg_img is not None:
            for ch in range(3):
                _img = img_stack[ch]
                means = (104, 117, 123)
                bg_img_ch = bg_img[:,:,2-ch]
                _img[invalid] = bg_img_ch[invalid] - means[2-ch]

        for ch in range(3):
            img_stack[ch] = torch.FloatTensor(img_stack[ch]).unsqueeze(0)
        img = torch.cat(img_stack, dim=0)
        
        return img, new_labels, index

    def _transform_label(self, bboxes, x, y):
        new_labels = []
        for x1,y1,x2,y2,c in bboxes:
            x1*=300
            x2*=300
            y1*=300
            y2*=300
            bbox = (x > x1) & (x < x2) & (y < y2) & (y > y1)
            (a, b) = np.where(bbox > 0)
            x1 = b.min()
            x2 = b.max()
            if x2-x1 == 1023:
                (_,b) = np.where(bbox[:,:512]>0)
                x3 = b.max()
                (_,b) = np.where(bbox[:,512:]>0)
                x4 = b.min() + 512
                if x3 >= 1023-x4:
                    x1 = x4-1024
                    x2 = x3
                else:
                    x1 = x4
                    x2 = 1024 + x3

            y1 = a.min()
            y2 = a.max()
            if y2-y1 == 511:
                (a,_) = np.where(bbox[:256,:]>0)
                y3 = a.max()
                (a,_) = np.where(bbox[256:,:]>0)
                y4 = a.min() + 256
                if y3 >= 511-y4:
                    y1 = y4-512
                    y2 = y3
                else:
                    y1 = y4
                    y2 = 512 + y3

            new_labels.append([x1/1024,y1/512,x2/1024,y2/512,c])

        return new_labels


from data import UCF24Detection, AnnotationTransform, BaseTransform, JHMDB

class OmniUCF24(OmniDataset):
    def __init__(self, root, image_set, transform=None, target_transform=None,
                 dataset_name='ucf24', input_type='rgb', full_test=False, use_npy=False, *args, **kwargs):
        if not use_npy:
            self.UCF24 = UCF24Detection(root, image_set, transform, target_transform,
                                        dataset_name, input_type, full_test)
            super(OmniUCF24, self).__init__(self.UCF24, *args, **kwargs)
        else:
            super(OmniUCF24, self).__init__(None, use_npy=use_npy, *args, **kwargs)

class OmniJHMDB(OmniDataset):
    def __init__(self, root, image_set, transform=None, target_transform=None, *args, **kwargs):
        self.JHMDB = JHMDB(root, image_set, transform, target_transform, split=1)
        super(OmniJHMDB, self).__init__(self.JHMDB, *args, **kwargs)

if __name__ == '__main__':
    import os
    import argparse
    from PIL import Image

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--idx', nargs='+', required=True,
                        help='image indices to demo')
    parser.add_argument('--out_dir', default='output/demo',
                        help='directory to output demo image')
    parser.add_argument('--dataset', default='OmniUCF24',
                        choices=['OmniUCF24','OmniJHMDB'],
                        help='which dataset to use')

    parser.add_argument('--fov', type=int, default=120,
                        help='fov of the tangent plane')
    parser.add_argument('--flip', action='store_true',
                        help='whether to apply random flip')
    parser.add_argument('--z_rotate', action='store_true',
                        help='whether to apply random panorama horizontal rotation')
    parser.add_argument('--y_rotate', action='store_true',
                        help='whether to apply random panorama vertical rotation')
    parser.add_argument('--fix_aug', action='store_true',
                        help='whether to apply random panorama vertical rotation')

    parser.add_argument('--ssd_dim', default=300, type=int,
                        help='Input Size for SSD') # only support 300 now
    parser.add_argument('--input_type', default='rgb', type=str,
                        help='INput tyep default rgb options are [rgb,brox,fastOF]')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    args.train_sets = 'train'
    args.means = (104, 117, 123)
    np.random.seed(112)

    if args.dataset == 'OmniUCF24':
        args.data_root = '/home/bo/research/dataset/ucf24/'
        dataset = OmniUCF24(args.data_root, 'test', BaseTransform(300, args.means),
                           AnnotationTransform(), input_type=args.input_type, full_test=True)
    elif args.dataset == 'OmniJHMDB':
        args.data_root = '/home/bo/research/dataset/jhmdb/'
        dataset = OmniJHMDB(args.data_root, 'test', BaseTransform(300, None),
                           AnnotationTransform())
    else:
        exit(0)

    print(len(dataset))
    from PIL import ImageDraw

    for idx in args.idx:
        idx = int(idx)
        path = os.path.join(args.out_dir, '%d.png' % idx)
        x, label, _ = dataset[idx]
        for ch in range(0,3):
            x[ch,:,:] += args.means[2-ch]

        print(path, label)
        img = Image.fromarray(x.permute(1, 2, 0).numpy().astype(np.uint8))
        draw = ImageDraw.Draw(img)
        for x1,y1,x2,y2,_ in label:
            x1*=1024
            x2*=1024
            y1*=512
            y2*=512
            x = (x1+x2)/2
            y = (y1+y2)/2
            w = x2-x1
            h = y2-y1
            draw.rectangle(((x1, y1), (x2, y2)), fill=None, outline ="black")

        img.save(path)

        # loaded_img = Image.open(path)

        # b,g,r = cv2.split(np.array(loaded_img))
        # frame_rgb = cv2.merge((r,g,b))
        # cv2.imshow('Edge sending window',frame_rgb)
        # cv2.waitKey(0)

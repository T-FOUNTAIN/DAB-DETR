import glob
import math
from pathlib import Path
import BboxToolkit as bt
import torch
import torch.utils.data as data
import torchvision
import datasets.dota_transforms as T
from PIL import Image
import os
import numpy as np
import util.misc
from matplotlib.patches import Rectangle

class DOTA:
    def __init__(self, dota_version, img_folder_path, ann_folder_path, ann_format):
        self.load_func = None
        if ann_format == 'txt':
            self.load_func = bt.load_dota
        elif ann_format == 'pkl':
            self.load_func = bt.load_pkl
        self.dota_version = dota_version
        self.img_folder_path = img_folder_path
        self.ann_folder_path = ann_folder_path

    def get_anns(self):
        contents, classes = self.load_func(img_dir=self.img_folder_path, ann_dir=self.ann_folder_path, classes=self.dota_version)
        return contents, classes

class DotaDetection(data.Dataset):
    def __init__(self, dota_version, img_folder, ann_folder, ann_format, transforms, large_scale_jitter,
                 image_set, dif_filter=False):
        self.dota = DOTA(dota_version, img_folder, ann_folder, ann_format)
        self.anns, self.classes = self.dota.get_anns()
        self.large_scale_jitter = large_scale_jitter
        self.image_set = image_set
        self.img_folder = img_folder
        self.ann_folder = ann_folder
        self._transforms = transforms
        self.prepare = Prepare(dif_filter)

    def __len__(self):
        return len(self.anns)

    def __getitem__(self, idx):
        target = self.anns[idx]
        filename = target['filename']
        img = Image.open(os.path.join(self.img_folder, filename)).convert('RGB')
        target = self.prepare(img.size, target)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        # if target["boxes"].shape[0]==0:
        #     return self.__getitem__((idx+1)%len)
        return img, target

# poly转xyhwr再转xyxyr，重构anno
# 根据difficult筛选box（可选）
# 暂时不筛选超出边界的bbox，
class Prepare(object):
    def __init__(self, dif_filter=False):
        self.dif_filter = dif_filter
    def __call__(self, wh, target):
        w, h = wh
        tgt = {}
        polys = target['ann']['bboxes'] # 八参数形式
        bboxes = bt.bbox2type(target['ann']['bboxes'], 'obb')
        hboxes, theta = bboxes[:, :4], bboxes[:, 4][...,None]
        # 将xywh转为xyxy形式
        hboxes[:, :2] -= hboxes[:, 2:]/2.0
        hboxes[:, 2:] = hboxes[:, :2] + hboxes[:, 2:]

        label = target['ann']['labels']
        if self.dif_filter == True:
            keep = np.where(target['ann']['diffs'] == 0)
            hboxes = hboxes[keep, :]
            theta = theta[keep, :]
            label = label[keep]
            polys = polys[keep, :]

        tgt['boxes'] = torch.as_tensor(hboxes)
        tgt['theta'] = torch.as_tensor(theta)
        tgt['polys'] = torch.as_tensor(polys)
        tgt['labels'] = torch.as_tensor(label)
        tgt['orig_size'] = torch.as_tensor([int(h), int(w)])
        tgt['size'] = torch.as_tensor([int(h), int(w)]) #未padding的长宽
        tgt['img_size'] = torch.as_tensor([int(h), int(w)])  # padding后的长宽
        tgt['rotation'] = torch.tensor([0, 1], dtype=torch.float32)
        img_id, patch_id = target['filename'].split('_')
        img_id = int(img_id[1:])
        patch_id = int(patch_id[:4])
        tgt['file_name'] = torch.tensor([img_id, patch_id])

        return tgt

def make_dota_transforms(image_set, large_scale_jitter):
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if image_set == 'train':
        if large_scale_jitter:
            return T.Compose([
                T.RandomHorizontalFlip(),
                T.RandomDistortion(0.5, 0.5, 0.5, 0.5),
                # # 返回最长边为800的图片+
                T.RotateAugmentation(),
                T.LargeScaleJitter(output_size=800, aug_scale_min=1, aug_scale_max=1),
                # # color jittering，以0.5概率随机改变图片对比度、亮度等
                normalize,
            ])
        else:
            return T.Compose([
                T.RandomHorizontalFlip(),
                T.RotateAugmentation(),
                T.RandomSelect(
                    # 短边缩放到scales列表中的任意一个值，最长边不超过800
                    T.RandomResize(scales, max_size=800),
                    T.Compose([
                        T.RandomResize([700, 800, 900]),
                        # 随机裁剪图片到h，w为384, 600的patch，我认为对已经split的dota可以不做,需要加此项时记得对mask也进行裁剪操作！
                        # T.RandomSizeCrop(384, 600), # 记得修改nested_tensor_list，修改mask
                        T.RandomResize(scales, max_size=800),
                    ])
                ),
                normalize,
            ])

    if image_set == 'val':
        if large_scale_jitter:
            return T.Compose([
                T.LargeScaleJitter(output_size=800, aug_scale_min=1.0, aug_scale_max=1.0),
                normalize,
            ])
        else:
            return T.Compose([
                T.RandomResize([800], max_size=800),
                normalize,
            ])

    raise ValueError(f'unknown {image_set}')

def build(image_set, args):
    root = Path(args.data_path)
    assert root.exists(), f'provided DOTA path {root} does not exist'
    if args.ann_format =='txt':
        PATHS = {
            "train": (root/ "train" / "images", root/ "train" / "labelTxt"),
            "val": (root/ "val" / "images", root/ "val"/ "labelTxt"),
        }
    elif args.ann_format == 'pkl':
        PATHS = {
            "train": (root / "train" / "images", root / "train" / "annfiles"/ "patch_annfile.pkl"),
            "val": (root / "val" / "images", root / "val" / "annfiles"/ "patch_annfile.pkl"),
        }
    else:
        raise ValueError(f'format of annotation file {args.ann_format} not supported')
    img_folder, ann_folder = PATHS[image_set]
    dataset = DotaDetection(
        dota_version = args.dataset_file,
        ann_format=args.ann_format,
        img_folder = img_folder,
        ann_folder = ann_folder,
        transforms = make_dota_transforms(image_set, args.large_scale_jitter),
        large_scale_jitter = args.large_scale_jitter,
        image_set = image_set,
        dif_filter = args.remove_difficult,
    )
    return dataset



if __name__ == '__main__':
    output_test_path = '/home/ttfang/dataset/test_dataloader/'
    dataset = DotaDetection(
        dota_version = 'dota1.0',
        img_folder = '/home/ttfang/dataset/split_dota_v1_800_05_pos/train/images',
        ann_folder = '/home/ttfang/dataset/split_dota_v1_800_05_pos/train/annfiles/patch_annfile.pkl',
        ann_format = 'pkl',
        # img_folder='/home/ttfang/dataset/DOTA-v1/train/images',
        # ann_folder='/home/ttfang/dataset/DOTA-v1/train/labelTxt',
        # ann_format = 'txt',
        transforms = make_dota_transforms("train", True),
        large_scale_jitter = True,
        image_set ='train',
    )

    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    dataloader = DataLoader(dataset, batch_size=1, collate_fn=util.misc.collate_fn, shuffle=False, num_workers=1)
    ite = iter(dataloader)
    files = glob.glob("/home/ttfang/dataset/test_dataloader/*.png")
    for file in files:
        os.remove(file)

    classes = dataset.classes
    for i in range(dataset.__len__()):
        data, target = next(ite)
        src, src_mask = data.decompose()
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1).repeat(src.shape[0], 1, src.shape[2], src.shape[3])
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1).repeat(src.shape[0], 1, src.shape[2], src.shape[3])
        src = (torch.mul(src, std)+mean)

        for j in range(len(target)):
            if target[j]['boxes'].shape[0]>0:
                h, w = target[j]['img_size']

                orig_box = target[j]['boxes'][0]* torch.tensor([w, h, w, h])
                r = target[j]['theta'][0]
                cos_r = torch.cos(r)
                sin_r = torch.sin(r)

                y = torch.arange(0, h, dtype=torch.float32)
                x = torch.arange(0, w, dtype=torch.float32)
                y, x = torch.meshgrid(y, x)
                grid = torch.stack((x, y), 2).unsqueeze(2).flatten(0, 1)

                R = torch.stack((cos_r, sin_r, -sin_r, cos_r), dim=-1).reshape(2, 2)
                gamma = torch.tensor([[orig_box[2].mul(orig_box[2])/4., torch.tensor([0.])], [torch.tensor([0.]), orig_box[3].mul(orig_box[3])/4]])
                Sigma_inv = torch.inverse(R.mm(gamma).mm(R.transpose(0, 1)))
                Sigma_inv_ext = Sigma_inv.unsqueeze(0).repeat(grid.shape[0], 1, 1) # [wh, 1, 2]

                miu = torch.tensor([orig_box[0], orig_box[1]], dtype=torch.float32).view(1, 1, 2).repeat(grid.shape[0], 1, 1)
                #gaussian_fx = torch.log((1./2*3.1415926*torch.sqrt(torch.det(Sigma)))*(torch.exp(-0.5*(grid-miu).bmm(torch.inverse(Sigma_ext)).bmm((grid-miu).transpose(1, 2)))))
                gaussian_fx =  (torch.exp(-0.5 * (grid - miu).bmm(Sigma_inv_ext).bmm((grid - miu).transpose(1, 2))))
                gaussian_fx_sigmoid = gaussian_fx.sigmoid()
                gaussian_fx_sigmoid = gaussian_fx_sigmoid.view(h, w)


                img_vis = src[j, ...]
                mask = src_mask[j, ...]
                img = np.array(img_vis.permute(1, 2, 0))

                img_vis = torchvision.transforms.ToPILImage()(img_vis)
                img_mask = torchvision.transforms.ToPILImage()(mask.float().unsqueeze(0).repeat(3,1,1))
                img_gau = torchvision.transforms.ToPILImage()(gaussian_fx_sigmoid.float().unsqueeze(0).repeat(3,1,1))

                img_vis.save(output_test_path+str(i)+'_'+str(j)+'.png')
                img_mask.save(output_test_path+str(i)+'_'+str(j)+'_mask.png')
                img_gau.save(output_test_path + str(i) + '_' + str(j) + '_gau_box.png')

                # 对box的可视化
                boxes = target[j]['boxes']
                theta = target[j]['theta']
                oboxes = np.array(torch.cat([boxes, theta], dim=-1))
                polys = bt.bbox2type(oboxes, 'poly') # 检验五参数
                #polys = np.array(target[j]['polys']) # 检验八参数

                height, width = img.shape[:2]
                ax, fig = bt.plt_init('', width, height)
                ax.imshow(img)

                text_vis = [classes[target[j]['labels'][ind]] for ind in range((target[j]['labels'].shape[0]))]
                for ind, box in enumerate(polys):
                    box = box*np.array([width, height, width, height, width, height, width, height])
                    bt.draw_poly(ax, box, texts=None, color='green')
                    ax.text(box[0], box[1], text_vis[ind], bbox=dict(facecolor='red', alpha=0.5))
                plt.show()

                img = bt.get_img_from_fig(fig, width, height)
                img = torchvision.transforms.ToPILImage()(img)

                img.save(output_test_path + str(i) + '_'+str(j)+'_with_bbox.png')

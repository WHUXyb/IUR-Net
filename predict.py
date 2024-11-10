import os, sys
sys.path.append('..')  # 将父级目录添加到导入搜索路径中 改  board and google库
# os.environ["CUDA_VISIBLE_DEVICES"]='2'
os.environ['KMP_DUPLICATE_LIB_OK']='True'  #解决正在重启内核bug
import argparse
import PIL.Image
import torch
from models.IURNet import IURNet
from torch.utils.data import DataLoader
import logging
from dataset.data_load import IURDataset
import time
from tqdm import tqdm
import random
import numpy as np
import cv2 as cv
import utils.measures as measures
import json
import utils.plot as plot
import warnings
from glob import glob
import os.path as osp
root = osp.dirname(osp.dirname(osp.abspath(__file__)))  #获取绝对路径

warnings.filterwarnings(action='ignore')


def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger

#反归一化操作
def denormalize(data, mean, std):
    mean = torch.Tensor(mean).type_as(data)
    std = torch.Tensor(std).type_as(data)
    return data.mul(std[..., None, None]).add(mean[..., None, None])


def get_confusion_matrix(label, pred, num_classes, ignore=-1):
    """
    calculate the confusion matrix by label and pred.
    """
    output = pred.cpu().numpy()#.transpose(0, 2, 3, 1)
    # pred_seg = np.asarray(np.argmax(output, axis=3), dtype=np.uint8)
    pred_seg = np.asarray(output, dtype=np.uint8)
    gt_seg = np.asarray(label.cpu().numpy(), dtype=np.int32)

    ignore_index = gt_seg != ignore
    gt_seg = gt_seg[ignore_index]
    pred_seg = pred_seg[ignore_index]

    index = (gt_seg*num_classes+pred_seg).astype(np.int32)

    label_count = np.bincount(index)
    confusion_matrix = np.zeros((num_classes, num_classes))

    for i_label in range(num_classes):
        for i_pred in range(num_classes):
            cur_index = i_label * num_classes + i_pred
            if cur_index < len(label_count):
                confusion_matrix[i_label,
                                 i_pred] = label_count[cur_index]
    return confusion_matrix


def get_data_loaders(args, Dataset):
    print('building dataloaders')
    dataset_val = Dataset(txt_file=args.test_data_txt,
                          mode='val',
                          use_reality_data=args.use_reality_data,
                          normal_imagenet=args.normal_imagenet)

    val_loader = DataLoader(dataset=dataset_val, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers)
    list_filename = [os.path.basename(p.split()[0]) for p in dataset_val.ids]
    return val_loader, list_filename


class Prediction(object):
    def __init__(self, args):
        self.args = args
        self.image_path = args.image_path
        self.label_uncheck_path = args.label_uncheck_path
        
        self.save_path2 = args.save_path2
        self.save_path_miss = args.save_path_miss
        self.save_path_error = args.save_path_error
        self.save_path_refine = args.save_path_refine
        self.save_path_prelabel = args.save_path_prelabel
        self. save_path_feature = args.save_path_feature
        self.best_performance = 0
        # self.opts = vars(args)
        # print(self.opts)  # 打印参数内容
        self.device = torch.device('cuda:{}'.format(args.GPU))
        self.network_name = args.network_name
        self.val_loader, self.list_filename = get_data_loaders(args, IURDataset)
        self.model = IURNet(in_channels=4, backbone='ResNet50',
                            with_mode='Ms-TripletAttention', with_aux=args.with_aux).to(self.device)

        if args.is_vis_map:
            self.vis_path = os.path.join(args.save_path1, self.network_name, 'vis_dirs')
            os.makedirs(self.vis_path, exist_ok=True)

        self.resume(args.weight_path)
    
    def resume(self, path):
        pre_weight = torch.load(path, map_location='cpu')

        model_dict = self.model.state_dict()
        num = 0
        for k, v in pre_weight.items():
            if 'model.' in k:
                k = k[6:]
            if k in model_dict.keys():
                model_dict[k] = v
                num += 1
        print(f'model-load-weight:{num}/{len(pre_weight.keys())}/{len(model_dict.keys())}')
        self.model.load_state_dict(model_dict)

    def image_mask_with_rgb(self, im, mask, rgb=[255, 0, 0]):
        for i in range(3):
            im[:, :, i][mask == 255] = rgb[i]
        return im
    
    
    def vis(self, im, label_uncheck, label_checked, label_res, pred_res):
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        mean = torch.Tensor(mean).type_as(im)
        std = torch.Tensor(std).type_as(im)
        im = im[0].mul(std[..., None, None]).add(mean[..., None, None])
        im = np.transpose(im.cpu().numpy() * 255, (1, 2, 0)).astype(np.uint8)
        label_uncheck = (label_uncheck[0].squeeze(0).detach().cpu().numpy() * 255).astype(np.uint8)
        label_checked = (label_checked[0].squeeze(0).detach().cpu().numpy() * 255).astype(np.uint8)
        label_res = (label_res[0].detach().cpu().numpy() * 255).astype(np.uint8)
        pred_res = (pred_res[0].detach().cpu().numpy() * 255).astype(np.uint8)
        label_res[label_res>127] = 255
        label_res[label_res<=127] = 0

        pred_res[pred_res > 127] = 255
        pred_res[pred_res <= 127] = 0
        ld_rgb = [0, 0, 255]  #蓝色
        err_rgb = [255, 0, 0]  #红色
        im_label_res = self.image_mask_with_rgb(im.copy(), label_res[0], rgb=ld_rgb)
        im_label_res = self.image_mask_with_rgb(im_label_res, label_res[1], rgb=err_rgb)

        im_pred_res = self.image_mask_with_rgb(im.copy(), pred_res[0], rgb=ld_rgb)
        im_pred_res = self.image_mask_with_rgb(im_pred_res, pred_res[1], rgb=err_rgb)
        return im, label_uncheck, label_checked, im_label_res, im_pred_res

    def validation(self, vis=False):
        self.model.eval()
        TP, Acc, PN, GN = [], [], [], []
        error = []
        for i_batch, (img, label_uncheck, label_checked, label_res) in tqdm(enumerate(self.val_loader)):
            img = img.to(self.device)
            label_uncheck = label_uncheck.to(self.device)
            label_checked = label_checked.to(self.device)
            label_res = label_res.to(self.device)
            with torch.no_grad():
                pred_res, _ = self.model(img, label_uncheck)  #

            if vis:
                im, label_uncheck, label_checked, im_label_res, im_pred_res = self.vis(img, label_uncheck, label_checked, label_res, pred_res)
                plot.visualize_list_data(list_data=[im, label_checked, label_uncheck, im_label_res, im_pred_res],
                                         M=2, N=3,
                                         list_titles=['im', 'label_checked', 'label_uncheck', 'im_label_res', 'im_pred_res'],
                                         save_dirs=os.path.join(self.vis_path, self.list_filename[i_batch]),
                                         show_data=False)
            tp, pn, gn, acc = measures.calculate_tp_pn_gn_accuracy(pred_res, label_res, threshold=0.5)
            er = measures.calculate_error(pred_res, threshold=0.5)
            TP.append(tp)
            PN.append(pn)
            GN.append(gn)
            Acc.append(acc)
            error.append(er)

        IoU = sum(TP) / (sum(PN) + sum(GN) - sum(TP) + 1e-9)
        Recall = sum(TP) / (sum(GN) + 1e-9)
        Precision = sum(TP) / (sum(PN) + 1e-9)
        F1 = 2 * Precision * Recall / (Precision + Recall)
        Accuracy = sum(Acc) / len(Acc)
        errors = sum(error) / (len(error))
        return IoU, Recall, Precision, Accuracy, F1, errors
    
    
    def predict(self, vis=False):
        start_time = time.time()
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        IoU, Recall, Precision, Accuracy, F1, errors = self.validation(vis=vis)
        print('IoU: %.3f%%' % (IoU.item() * 100))
        print('Recall: %.3f%%' % (Recall.item() * 100))
        print('Precision: %.3f%%' % (Precision.item() * 100))
        print('Accuracy: %.3f%%' % (Accuracy.item() * 100))
        print('F1: %.3f%%' % (F1.item() * 100))
        print('error: %.3f%%' % (errors.item() * 100))
        duration1 = time.time() - start_time
        print('Val-used-time:', duration1)
    
    
    def vis_results(self):

        # os.makedirs(self.save_path2, exist_ok=True)
        # os.makedirs(self.save_path_miss, exist_ok=True)
        # os.makedirs(self.save_path_error, exist_ok=True)
        # os.makedirs(self.save_path_refine, exist_ok=True)
        os.makedirs(self.save_path_prelabel, exist_ok=True)
        # os.makedirs(self.save_path_feature, exist_ok=True)
        
        # self.resume(self.args.weight_path)

        self.model.eval()
        for i_batch, (img, label_uncheck, label_checked, label_res) in tqdm(enumerate(self.val_loader)):
            img = img.to(self.device)
            label_uncheck = label_uncheck.to(self.device)

            with torch.no_grad():
                # pred_res = self.model(img, label_uncheck)
                pred_res, fea_re = self.model(img, label_uncheck)

            #输出辅助分支分割结果
            pred_label = (pred_res[1].squeeze().detach().cpu().numpy() * 255).astype(np.uint8)
            # pred_label=pred_label[0,0,:,:]
            pred_label[pred_label > 127] = 255
            pred_label[pred_label <= 127] = 0
            
            pred_res = (pred_res[0].detach().cpu().numpy() * 255).astype(np.uint8)
            pred_res[pred_res > 127] = 255
            pred_res[pred_res <= 127] = 0
            
            basename = self.list_filename[i_batch]
            image = np.asarray(PIL.Image.open(osp.join(self.image_path, basename)))
            label_uncheck = cv.imread(osp.join(self.label_uncheck_path, basename), 0)
            res_miss=pred_res[0,0,:,:]
            res_error=pred_res[0,1,:,:]
            refinelabel = label_uncheck + res_miss - res_error
            refinelabel[refinelabel == 1] = 0
            refinelabel[refinelabel == 254] = 255

            refinelabel = cv.morphologyEx(refinelabel, cv.MORPH_OPEN, np.ones((3,3), np.uint8))

            refinelabel = cv.morphologyEx(refinelabel, cv.MORPH_CLOSE, np.ones((3,3), np.uint8))
            
            # cv.imwrite(osp.join(self.save_path_refine, basename), refinelabel)
            # cv.imwrite(osp.join(self.save_path_miss, basename), res_miss)
            # cv.imwrite(osp.join(self.save_path_error, basename), res_error)
            cv.imwrite(osp.join(self.save_path_prelabel, basename), pred_label)
            # # 确保tensor在cpu上并且已经从梯度计算中脱离
            # fea_re = fea_re.cpu().detach()
            # torch.save(fea_re, os.path.join(self.save_path_feature, f"{basename}_fea_re.pt"))
            
            # plot.vis_image_with_questioned_pixels(res_miss=pred_res[0,0,:,:],
            #                                       res_error=pred_res[0,1,:,:],
            #                                       label_uncheck=label_uncheck,
            #                                       image=image,
            #                                       save_tif_path=osp.join(self.save_path2, basename),
            #                                       show_data=False)


if __name__ == '__main__':
    start_time = time.time()
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--seed', type=int, default=666)
    parser.add_argument('--use_reality_data', type=bool, default=True)
    parser.add_argument('--normal_imagenet', type=bool, default=True)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--with_aux', type=bool, default=True)
    parser.add_argument('--network_name', type=str, default='IURNet')
    parser.add_argument('--GPU', type=int, default=0)
    parser.add_argument('--is_vis_map', type=bool, default=False)

    
    parser.add_argument("--image_path", type=str, default=r'C:\Users\Xiong\Desktop\RefineNet\buildingsdata\test_buildings\image')
    parser.add_argument("--label_uncheck_path", type=str, default=r'C:\Users\Xiong\Desktop\RefineNet\buildingsdata\test_buildings\label_uncheck')
    parser.add_argument('--test_data_txt', type=str, default=r'C:\Users\Xiong\Desktop\RefineNet\buildingsdata\txt\test.txt')
    
    parser.add_argument('--save_path1', type=str, default=r'C:\Users\Xiong\Desktop\RefineNet\noisy_ratio\result_pre_vis1_wei36_test')
    parser.add_argument("--save_path2", type=str, default=r'C:\Users\Xiong\Desktop\RefineNet\noisy_ratio\result_pre_vis2_wei36_test')
    parser.add_argument("--save_path_miss", type=str, default=r'C:\Users\Xiong\Desktop\RefineNet\noisy_ratio\result_pre_vis2_wei36_miss_test')
    parser.add_argument("--save_path_error", type=str, default=r'C:\Users\Xiong\Desktop\RefineNet\noisy_ratio\result_pre_vis2_wei36_error_test')
    parser.add_argument("--save_path_feature", type=str, default=r'C:\Users\Xiong\Desktop\RefineNet\noisy_ratio\result_pre_vis2_wei36_feature_test')
    parser.add_argument("--save_path_prelabel", type=str, default=r'C:\Users\Xiong\Desktop\RefineNet\noisy_ratio\60\result_pre_vis2_wei36_prelabel_test')
    
    parser.add_argument("--save_path_refine", type=str, default=r'C:\Users\Xiong\Desktop\RefineNet\noisy_ratio\10\result_pre_vis2_wei36_detect+replace_test_base')
    parser.add_argument('--weight_path', type=str, default=r'C:\Users\xiong\Desktop\RefineNet\trained_weights\wIUR\IURnet_wIUR_hrnet.pth')

    args = parser.parse_args()
    
    trainer = Prediction(args)
    trainer.vis_results()
    end_time = time.time()
    inference_time = end_time - start_time
    print(f"推理时间: {inference_time} 秒")
    # trainer.predict(vis=True)  #predict→validation→vis

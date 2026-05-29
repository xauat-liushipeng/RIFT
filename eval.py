import numpy as np
import os
import argparse
import logging
import glob
import cv2


def cal_global_acc(pred, gt):
    h, w = gt.shape
    return [np.sum(pred == gt), float(h * w)]


def get_statistics_seg(pred, gt, num_cls=2):
    h, w = gt.shape
    statistics = []
    for i in range(num_cls):
        tp = np.sum((pred == i) & (gt == i))
        fp = np.sum((pred == i) & (gt != i))
        fn = np.sum((pred != i) & (gt == i))
        statistics.append([tp, fp, fn])
    return statistics


def get_statistics_prf(pred, gt):
    tp = np.sum((pred == 1) & (gt == 1))
    fp = np.sum((pred == 1) & (gt == 0))
    fn = np.sum((pred == 0) & (gt == 1))
    return [tp, fp, fn]


def segment_metrics(pred_list, gt_list, num_cls=2):
    global_accuracy_cur = []
    statistics = []
    
    for pred, gt in zip(pred_list, gt_list):
        gt_img = (gt / 255).astype('uint8')
        pred_img = (pred / 255).astype('uint8')
        global_accuracy_cur.append(cal_global_acc(pred_img, gt_img))
        statistics.append(get_statistics_seg(pred_img, gt_img, num_cls))
    
    global_acc = np.sum([v[0] for v in global_accuracy_cur]) / np.sum([v[1] for v in global_accuracy_cur])
    counts = []
    for i in range(num_cls):
        tp = np.sum([v[i][0] for v in statistics])
        fp = np.sum([v[i][1] for v in statistics])
        fn = np.sum([v[i][2] for v in statistics])
        
        counts.append([tp, fp, fn])
    
    mean_acc = np.sum([v[0] / (v[0] + v[2]) for v in counts]) / num_cls
    mean_iou_acc = np.sum([v[0] / (np.sum(v)) for v in counts]) / num_cls
    
    return global_acc, mean_acc, mean_iou_acc


def prf_metrics(pred_list, gt_list):
    statistics = []
    
    for pred, gt in zip(pred_list, gt_list):
        gt_img = (gt / 255).astype('uint8')
        max_value = np.max(pred)
        pred_img = np.zeros_like(pred, dtype='uint8') if max_value <= 0 else ((pred / max_value) > 0.5).astype('uint8')
        statistics.append(get_statistics_prf(pred_img, gt_img))
    
    tp = np.sum([v[0] for v in statistics])
    fp = np.sum([v[1] for v in statistics])
    fn = np.sum([v[2] for v in statistics])
    print("tp:{}, fp:{}, fn:{}".format(tp, fp, fn))
    p_acc = 1.0 if tp == 0 and fp == 0 else tp / (tp + fp)
    r_acc = tp / (tp + fn)
    f_acc = 2 * p_acc * r_acc / (p_acc + r_acc)
    return p_acc, r_acc, f_acc


def cal_prf_metrics(pred_list, gt_list, thresh_step=0.01):
    final_accuracy_all = []
    for thresh in np.arange(0.0, 1.0, thresh_step):
        statistics = []
        for pred, gt in zip(pred_list, gt_list):
            gt_img = (gt / 255).astype('uint8')
            pred_img = (pred / 255 > thresh).astype('uint8')
            statistics.append(get_statistics(pred_img, gt_img))
        tp = np.sum([v[0] for v in statistics])
        fp = np.sum([v[1] for v in statistics])
        fn = np.sum([v[2] for v in statistics])
        
        p_acc = 1.0 if tp == 0 and fp == 0 else tp / (tp + fp)
        r_acc = tp / (tp + fn)
        final_accuracy_all.append([thresh, p_acc, r_acc, 2 * p_acc * r_acc / (p_acc + r_acc)])
    
    return final_accuracy_all


def thred_half(src_img_list, tgt_img_list):
    Precision, Recall, F_score = prf_metrics(src_img_list, tgt_img_list)
    Global_Accuracy, Class_Average_Accuracy, Mean_IOU = segment_metrics(src_img_list, tgt_img_list)
    print("Global Accuracy:{}, Class Average Accuracy:{}, Mean IOU:{}, Precision:{}, Recall:{}, F score:{}".format(
        Global_Accuracy, Class_Average_Accuracy, Mean_IOU, Precision, Recall, F_score))


def get_statistics(pred, gt):
    tp = np.sum((pred == 1) & (gt == 1))
    fp = np.sum((pred == 1) & (gt == 0))
    fn = np.sum((pred == 0) & (gt == 1))
    return [tp, fp, fn]


def cal_OIS_metrics(pred_list, gt_list, thresh_step=0.01):
    final_F1_list = []
    for pred, gt in zip(pred_list, gt_list):
        p_acc_list = []
        r_acc_list = []
        F1_list = []
        for thresh in np.arange(0.0, 1.0, thresh_step):
            gt_img = (gt / 255).astype('uint8')
            pred_img = (pred / 255 > thresh).astype('uint8')
            tp, fp, fn = get_statistics(pred_img, gt_img)
            p_acc = 1.0 if tp == 0 and fp == 0 else tp / (tp + fp)
            if tp + fn == 0:
                r_acc = 0
            else:
                r_acc = tp / (tp + fn)
            if p_acc + r_acc == 0:
                F1 = 0
            else:
                F1 = 2 * p_acc * r_acc / (p_acc + r_acc)
            
            p_acc_list.append(p_acc)
            r_acc_list.append(r_acc)
            F1_list.append(F1)
        
        assert len(p_acc_list) == 100, "p_acc_list is not 100"
        assert len(r_acc_list) == 100, "r_acc_list is not 100"
        assert len(F1_list) == 100, "F1_list is not 100"
        
        max_F1 = np.max(np.array(F1_list))
        final_F1_list.append(max_F1)
    
    final_F1 = np.sum(np.array(final_F1_list)) / len(final_F1_list)
    return final_F1


def cal_ODS_metrics(pred_list, gt_list, thresh_step=0.01):
    save_data = {
        "ODS": [],
    }
    final_ODS = []
    for thresh in np.arange(0.0, 1.0, thresh_step):
        ODS_list = []
        for pred, gt in zip(pred_list, gt_list):
            gt_img = (gt / 255).astype('uint8')
            pred_img = (pred / 255 > thresh).astype('uint8')
            tp, fp, fn = get_statistics(pred_img, gt_img)
            # calculate precision
            p_acc = 1.0 if tp == 0 and fp == 0 else tp / (tp + fp)
            if tp + fn == 0:
                r_acc = 0
            else:
                r_acc = tp / (tp + fn)
            if p_acc + r_acc == 0:
                F1 = 0
            else:
                F1 = 2 * p_acc * r_acc / (p_acc + r_acc)
            ODS_list.append(F1)
        
        ave_F1 = np.mean(np.array(ODS_list))
        final_ODS.append(ave_F1)
    ODS = np.max(np.array(final_ODS))
    return ODS


def cal_mIoU_metrics(pred_list, gt_list, thresh_step=0.05, pred_imgs_names=None, gt_imgs_names=None):
    num_pairs = len(pred_list)
    pred_norm_list = [pred / 255 for pred in pred_list]
    gt_bin_list = [(gt / 255).astype('uint8') for gt in gt_list]
    iou_buffer = np.empty(num_pairs, dtype=np.float64)
    best_miou = -np.inf
    
    for thresh in np.arange(0.0, 1.0, thresh_step):
        for idx, (pred_norm, gt_img) in enumerate(zip(pred_norm_list, gt_bin_list)):
            pred_img = (pred_norm > thresh).astype('uint8')
            TP = np.sum((pred_img == 1) & (gt_img == 1))
            TN = np.sum((pred_img == 0) & (gt_img == 0))
            FP = np.sum((pred_img == 1) & (gt_img == 0))
            FN = np.sum((pred_img == 0) & (gt_img == 1))
            if (FN + FP + TP) <= 0:
                iou_buffer[idx] = 0.0
            else:
                iou_1 = TP / (FN + FP + TP)
                iou_0 = TN / (FN + FP + TN)
                iou_buffer[idx] = (iou_1 + iou_0) / 2
        ave_iou = np.mean(iou_buffer)
        if ave_iou > best_miou:
            best_miou = ave_iou
    return best_miou


def imread(path, load_size=0, load_mode=cv2.IMREAD_GRAYSCALE, convert_rgb=False, thresh=-1):
    im = cv2.imread(path, load_mode)
    if convert_rgb:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    if load_size > 0:
        im = cv2.resize(im, (load_size, load_size), interpolation=cv2.INTER_CUBIC)
    if thresh > 0:
        _, im = cv2.threshold(im, thresh, 255, cv2.THRESH_BINARY)
    return im


def get_image_pairs(data_dir, suffix_gt='real_B', suffix_pred='fake_B'):
    gt_list = glob.glob(os.path.join(data_dir, '*{}.png'.format(suffix_gt)))
    pred_list = [ll.replace(suffix_gt, suffix_pred) for ll in gt_list]
    assert len(gt_list) == len(pred_list)
    pred_imgs, gt_imgs = [], []
    pred_imgs_names, gt_imgs_names = [], []
    for pred_path, gt_path in zip(pred_list, gt_list):
        pred_imgs.append(imread(pred_path))
        gt_imgs.append(imread(gt_path, thresh=127))
        pred_imgs_names.append(pred_path)
        gt_imgs_names.append(gt_path)
    return pred_imgs, gt_imgs, pred_imgs_names, gt_imgs_names


def evaluate_directory(results_dir, thresh_step=0.01):
    suffix_gt = "lab"
    suffix_pred = "pre"
    src_img_list, tgt_img_list, pred_imgs_names, gt_imgs_names = get_image_pairs(results_dir, suffix_gt, suffix_pred)
    if not src_img_list:
        raise FileNotFoundError(f"No prediction/label pairs found in: {results_dir}")
    assert len(src_img_list) == len(tgt_img_list)

    final_accuracy_all = np.array(cal_prf_metrics(src_img_list, tgt_img_list, thresh_step=thresh_step))
    precision_list = final_accuracy_all[:, 1]
    recall_list = final_accuracy_all[:, 2]
    f1_list = final_accuracy_all[:, 3]
    miou = cal_mIoU_metrics(
        src_img_list,
        tgt_img_list,
        thresh_step=thresh_step,
        pred_imgs_names=pred_imgs_names,
        gt_imgs_names=gt_imgs_names,
    )
    ods = cal_ODS_metrics(src_img_list, tgt_img_list, thresh_step=thresh_step)
    ois = cal_OIS_metrics(src_img_list, tgt_img_list, thresh_step=thresh_step)
    return {
        "mIoU": float(miou),
        "ODS": float(ods),
        "OIS": float(ois),
        "F1": float(np.max(f1_list)),
        "Precision": float(np.max(precision_list)),
        "Recall": float(np.max(recall_list)),
    }


def eval(log_eval, results_dir, epoch):
    log_eval.info(results_dir)
    log_eval.info("checkpoints -> " + results_dir)
    metrics = evaluate_directory(results_dir)
    for key, value in metrics.items():
        log_eval.info(f"{key} -> {value}")
    log_eval.info("eval finish!")
    return {"epoch": epoch, "mIoU": metrics["mIoU"]}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate RIFT prediction results")
    parser.add_argument(
        "--results_dir",
        type=str,
        default="./results_test",
        help="Directory containing <name>_lab.png and <name>_pre.png files.",
    )
    parser.add_argument(
        "--thresh_step",
        type=float,
        default=0.01,
        help="Threshold step for PR/F1/OIS/ODS and mIoU sweeping.",
    )
    args = parser.parse_args()
    metrics = evaluate_directory(args.results_dir, thresh_step=args.thresh_step)
    logging.info(args.results_dir)
    for key, value in metrics.items():
        print(f"{key} -> {value}")
    print("eval finish!")




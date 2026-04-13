import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

# 加载 YOLO11-Seg 模型
model = YOLO('best.pt')

# 输入输出目录
source_dir = Path('source_img')
output_dir = Path('output_img')
output_dir.mkdir(exist_ok=True)

# 遍历 source_img 中的所有图片
image_exts = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
image_files = [f for f in source_dir.iterdir() if f.suffix.lower() in image_exts]

if not image_files:
    print('source_img 中没有找到图片文件')
    exit()


def get_bottom_quarter_mask(full_mask, h, w):
    """
    基于最小旋转矩形的边识别底边，求掩膜底部 1/4 区域（平行四边形切割）。
    返回 (bottom_mask, BL, BR)。
    """
    # 找到掩膜轮廓
    contours, _ = cv2.findContours(full_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros((h, w), dtype=np.uint8), None, None

    # 取最大轮廓
    cnt = max(contours, key=cv2.contourArea)

    # 求最小面积旋转矩形，获取四个角点
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)  # 4 个角点，shape (4, 2)，相邻角点共享一条边

    # 基于边识别底边：计算 4 条边的平均 y，取最大的为底边
    edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
    avg_ys = [(box[i][1] + box[j][1]) / 2.0 for i, j in edges]
    bottom_edge_idx = int(np.argmax(avg_ys))
    bi, bj = edges[bottom_edge_idx]

    # 底边两端点按 x 排序为 BL（左）、BR（右）
    if box[bi][0] <= box[bj][0]:
        BL, BR = box[bi].copy(), box[bj].copy()
        bl_idx, br_idx = bi, bj
    else:
        BL, BR = box[bj].copy(), box[bi].copy()
        bl_idx, br_idx = bj, bi

    # 通过邻接关系找到 TL（与 BL 相邻且非 BR）和 TR（与 BR 相邻且非 BL）
    neighbors_bl = [(bl_idx - 1) % 4, (bl_idx + 1) % 4]
    TL = box[[n for n in neighbors_bl if n != br_idx][0]]

    neighbors_br = [(br_idx - 1) % 4, (br_idx + 1) % 4]
    TR = box[[n for n in neighbors_br if n != bl_idx][0]]

    # 在 75% 处沿侧边插值得到切割线端点（平行四边形）
    CL = TL + 0.75 * (BL - TL)
    CR = TR + 0.75 * (BR - TR)

    # 构造底部 1/4 四边形：CL -> CR -> BR -> BL
    cut_quad = np.array([CL, CR, BR, BL], dtype=np.int32)

    # 绘制切割四边形为二值 mask
    cut_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(cut_mask, [cut_quad], 255)

    # 与原始掩膜取交集
    bottom_mask = cv2.bitwise_and(full_mask, cut_mask)

    return bottom_mask, BL, BR


def get_top_edge_points(bottom_mask):
    """
    从 bottom_mask 的上边缘逐列扫描，取掩膜实际像素范围内的列，
    保留顶部边缘点。
    """
    cols_with_pixels = np.any(bottom_mask > 0, axis=0)
    x_indices = np.where(cols_with_pixels)[0]
    if len(x_indices) == 0:
        return []
    x_left = int(x_indices[0])
    x_right = int(x_indices[-1])

    edge_points = []
    for x in range(x_left, x_right + 1):
        col = bottom_mask[:, x]
        ys = np.where(col > 0)[0]
        if len(ys) > 0:
            edge_points.append((x, int(ys[0])))
    return edge_points


def fit_line_and_sample(edge_points, BL, BR, num_points=10):
    """
    固定底边方向拟合：方向锁定为 BR-BL，仅拟合法向偏移量（中位数），
    用自适应离群点过滤确定线段端点，均匀采样 num_points 个点。
    返回 (pt_left, pt_right, sample_points) 或 (None, None, [])
    """
    if len(edge_points) < 2 or BL is None or BR is None:
        return None, None, []

    pts = np.array(edge_points, dtype=np.float64)
    BL = BL.astype(np.float64)
    BR = BR.astype(np.float64)

    # 底边方向向量，归一化
    d = BR - BL
    d_len = np.linalg.norm(d)
    if d_len < 1e-6:
        return None, None, []
    d_hat = d / d_len

    # 法向量（指向上方）
    n_hat = np.array([-d_hat[1], d_hat[0]])
    if n_hat[1] > 0:  # 确保指向上方（y 减小方向）
        n_hat = -n_hat

    # 每个边缘点相对 BL 的偏移
    deltas = pts - BL  # (N, 2)

    # 法向投影（到底边的法向距离）
    t = deltas @ n_hat  # (N,)
    # 底边方向投影
    s = deltas @ d_hat  # (N,)

    # 法向偏移取中位数（鲁棒）
    t_med = float(np.median(t))

    # 自适应离群点过滤（基于法向残差）
    abs_residuals = np.abs(t - t_med)
    median_res = np.median(abs_residuals)
    mad = np.median(np.abs(abs_residuals - median_res))
    threshold = max(median_res + 3.0 * max(mad, 1.0), 5.0)

    inlier_mask = abs_residuals <= threshold
    s_inliers = s[inlier_mask]

    if len(s_inliers) < 2:
        s_inliers = s

    s_min = float(s_inliers.min())
    s_max = float(s_inliers.max())

    # 端点
    pt_left = BL + s_min * d_hat + t_med * n_hat
    pt_right = BL + s_max * d_hat + t_med * n_hat

    # 均匀采样
    sample_pts = []
    for i in range(num_points):
        frac = i / (num_points - 1)
        pt = pt_left + frac * (pt_right - pt_left)
        sample_pts.append((int(round(pt[0])), int(round(pt[1]))))

    return pt_left, pt_right, sample_pts


for image_path in image_files:
    img = cv2.imread(str(image_path))
    if img is None:
        print(f'无法读取图片: {image_path.name}，跳过')
        continue

    h, w = img.shape[:2]
    stem = image_path.stem
    suffix = image_path.suffix
    output_name = f'{stem}_mask{suffix}'
    output_path = output_dir / output_name
    mask_txt_path = output_dir / f'{stem}_mask.txt'

    # 推理
    results = model(str(image_path))

    # 获取分割掩膜
    result = results[0]
    if result.masks is not None:
        masks = result.masks.xy
    else:
        print(f'{image_path.name}: 未检测到分割掩膜，跳过')
        continue

    # 绘制掩膜
    img_mask = img.copy()
    overlay_green = img.copy()
    overlay_red = img.copy()

    bottom_quarter_contours = []
    all_bottom_masks = []

    for mask_pts in masks:
        pts = mask_pts.astype(np.int32)

        # 绿色填充完整掩膜
        cv2.fillPoly(overlay_green, [pts], color=(0, 255, 0))
        cv2.polylines(img_mask, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

        # 生成完整掩膜的二值 mask
        full_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(full_mask, [pts], 255)

        # 基于旋转矩形求底部 1/4
        bottom_mask, BL, BR = get_bottom_quarter_mask(full_mask, h, w)
        all_bottom_masks.append((bottom_mask, BL, BR))

        # 提取下方 1/4 的轮廓
        contours, _ = cv2.findContours(bottom_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            cv2.fillPoly(overlay_red, [cnt], color=(0, 0, 255))
            cv2.polylines(img_mask, [cnt], isClosed=True, color=(0, 0, 255), thickness=2)
            bottom_quarter_contours.append(cnt)

    img_mask = cv2.addWeighted(overlay_green, 0.3, img_mask, 0.7, 0)
    img_mask = cv2.addWeighted(overlay_red, 0.4, img_mask, 0.6, 0)

    # 对每个底部 1/4 mask，拟合上边缘直线并均匀采样 10 个点
    points_txt_path = output_dir / f'{stem}_points.txt'
    all_sample_points = []
    for i, (bm, BL, BR) in enumerate(all_bottom_masks):
        edge_pts = get_top_edge_points(bm)
        pt_left, pt_right, sample_pts = fit_line_and_sample(edge_pts, BL, BR, num_points=10)
        if pt_left is None:
            all_sample_points.append([])
            continue
        all_sample_points.append(sample_pts)
        # 绘制拟合直线（黄色）
        cv2.line(img_mask, (int(round(pt_left[0])), int(round(pt_left[1]))),
                 (int(round(pt_right[0])), int(round(pt_right[1]))), (0, 255, 255), 2)
        # 在图像上绘制采样点（蓝色圆点）
        for px, py in sample_pts:
            cv2.circle(img_mask, (px, py), 5, (255, 0, 0), -1)

    # 保存采样点坐标
    with open(points_txt_path, 'w') as f:
        for i, sample_pts in enumerate(all_sample_points):
            f.write(f'basket_{i}_top_edge_points:\n')
            for px, py in sample_pts:
                f.write(f'{px},{py}\n')

    # 保存掩膜坐标
    with open(mask_txt_path, 'w') as f:
        for i, mask_pts in enumerate(masks):
            coords = mask_pts.astype(int)
            coord_str = ';'.join([f'{x},{y}' for x, y in coords])
            f.write(f'basket_{i}_full: {coord_str}\n')
        for i, cnt in enumerate(bottom_quarter_contours):
            pts_out = cnt.reshape(-1, 2).astype(int)
            coord_str = ';'.join([f'{x},{y}' for x, y in pts_out])
            f.write(f'basket_{i}_bottom25: {coord_str}\n')

    # 保存带掩膜的图像
    cv2.imwrite(str(output_path), img_mask)
    print(f'{image_path.name} -> {output_name} 完成')

print('所有图片处理完成')

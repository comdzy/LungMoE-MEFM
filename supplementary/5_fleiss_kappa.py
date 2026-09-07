"""
Fleiss K 一致性分析 (用于论文 "表 X: 9 个征象的 Fleiss K")

数据源:
  1. 原始 LIDC XML (首选, 路径在 DS_ROOT)
     - 每个 radiologist 对每个 nodule 的每个 attribute 给一个 1-6 的整数评级
  2. 预抽取的 per-rater JSON (备选, 路径在 --ratings_json)
     - 格式: {nodule_id: {reader_id: {attr: rating, ...}, ...}, ...}

Fleiss K 公式:
  - N 个 nodule, k 个 reader, q 个类别
  - n_ij = nodule i 在类别 j 上的 reader 数
  - P_i = (sum_j n_ij^2 - n_i) / (n_i * (n_i - 1))   # 1 nodule 的 agreement
  - P_bar = mean_i(P_i)
  - p_j = sum_i n_ij / (N * k)
  - P_e = sum_j p_j^2
  - K = (P_bar - P_e) / (1 - P_e)
"""
import os, sys, json, re, argparse
import xml.etree.ElementTree as ET
from collections import defaultdict
import numpy as np

XMLNS = '{http://www.nih.gov}'
PATIENT_RE = re.compile(r'(LIDC-IDRI-\d+)')

# 9 个属性
ATTR_NAMES = ['malignancy', 'subtlety', 'internalStructure', 'calcification',
              'sphericity', 'margin', 'lobulation', 'spiculation', 'texture']

# LIDC 类别数: Malignancy=5 类 (1-5), 其它=6 类 (1-6)
ATTR_CLASSES = {
    'malignancy': 5, 'subtlety': 6, 'internalStructure': 4, 'calcification': 6,
    'sphericity': 5, 'margin': 5, 'lobulation': 5, 'spiculation': 6, 'texture': 5
}


def is_lidc_ct_xml(xml_path):
    try:
        with open(xml_path, 'rb') as f:
            head = f.read(4096).decode('utf-8', errors='ignore')
        return 'LidcReadMessage' in head and 'IdriReadMessage' not in head
    except:
        return False


def parse_xml_ratings(xml_path, nodule_key):
    """提取该 XML 中对该 nodule_key 的 9 属性评级.
       配对 key: (patient_id, round_x_20, round_y_20)"""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        session = root.find(XMLNS + 'readingSession')
        if session is None: return None
        rad_node = session.find(XMLNS + 'servicingRadiologistID')
        rad_id = rad_node.text if (rad_node is not None and rad_node.text is not None) else 'unknown'
        m = PATIENT_RE.search(xml_path)
        pid = m.group(1) if m else 'unknown'
        for n in session.findall(XMLNS + 'unblindedReadNodule'):
            ch = n.find(XMLNS + 'characteristics')
            if ch is None: continue
            xs_all, ys_all = [], []
            for roi in n.findall(XMLNS + 'roi'):
                edges = roi.findall('.//ns:edgeMap', {'ns': 'http://www.nih.gov'})
                for e in edges:
                    xs_all.append(int(e.find(XMLNS + 'xCoord').text))
                    ys_all.append(int(e.find(XMLNS + 'yCoord').text))
            if not xs_all: continue
            cx = (min(xs_all)+max(xs_all)) / 2
            cy = (min(ys_all)+max(ys_all)) / 2
            key = (pid, int(cx) // 20 * 20, int(cy) // 20 * 20)
            if key != nodule_key: continue
            attrs = {}
            for a in ATTR_NAMES:
                node = ch.find(XMLNS + a)
                if node is not None and node.text is not None:
                    try: attrs[a] = int(node.text)
                    except: pass
            return (rad_id, attrs)
    except:
        return None
    return None


def build_ratings_from_xml(ds_root):
    """扫描整个 ds_root, 收集所有 (nodule_key) × (reader) × (attr) 评级.
       返回: {nodule_key: {reader_id: {attr: rating}}}"""
    print(f'扫描 {ds_root} ...', flush=True)
    nodule_to_ratings = defaultdict(lambda: defaultdict(dict))
    skipped_cxr = 0
    n_files = 0
    for root, _, files in os.walk(ds_root):
        for fn in files:
            if not fn.lower().endswith('.xml'): continue
            xml_path = os.path.join(root, fn)
            if not is_lidc_ct_xml(xml_path):
                skipped_cxr += 1
                continue
            n_files += 1
            tree = ET.parse(xml_path)
            xroot = tree.getroot()
            session = xroot.find(XMLNS + 'readingSession')
            if session is None: continue
            rad_node = session.find(XMLNS + 'servicingRadiologistID')
            rad_id = rad_node.text if (rad_node is not None and rad_node.text is not None) else 'unknown'
            m = PATIENT_RE.search(xml_path)
            pid = m.group(1) if m else 'unknown'
            for n in session.findall(XMLNS + 'unblindedReadNodule'):
                ch = n.find(XMLNS + 'characteristics')
                if ch is None: continue
                xs_all, ys_all = [], []
                for roi in n.findall(XMLNS + 'roi'):
                    edges = roi.findall('.//ns:edgeMap', {'ns': 'http://www.nih.gov'})
                    for e in edges:
                        xs_all.append(int(e.find(XMLNS + 'xCoord').text))
                        ys_all.append(int(e.find(XMLNS + 'yCoord').text))
                if not xs_all: continue
                cx = (min(xs_all)+max(xs_all)) / 2
                cy = (min(ys_all)+max(ys_all)) / 2
                key = (pid, int(cx) // 20 * 20, int(cy) // 20 * 20)
                for a in ATTR_NAMES:
                    node = ch.find(XMLNS + a)
                    if node is not None and node.text is not None:
                        try: nodule_to_ratings[key][rad_id][a] = int(node.text)
                        except: pass
    print(f'扫描 {n_files} XML (跳过 CXR {skipped_cxr}), {len(nodule_to_ratings)} unique nodule', flush=True)
    return dict(nodule_to_ratings)


def fleiss_kappa(ratings, n_categories):
    """
    ratings: list of lists, each inner list is k readers' ratings for one item
             e.g. [[3,3,1,2], [4,4,4,1], ...]
    n_categories: q (类别数)
    Returns: K (float)
    """
    N = len(ratings)
    if N < 2: return float('nan')
    # k = number of raters per item (assume all same)
    ks = [len(r) for r in ratings]
    if len(set(ks)) > 1:
        # 不同 nodule 评人数不同: 分别计算
        # 简化: 取众数 k, 不一致者按 n_i 计入 P_i
        from collections import Counter
        k = Counter(ks).most_common(1)[0][0]
        # 只用 k-reader 的
        ratings = [r for r in ratings if len(r) == k]
        N = len(ratings)
        if N < 2: return float('nan')
    else:
        k = ks[0]
    if k < 2: return float('nan')

    # 每 nodule P_i
    P_i_list = []
    for r in ratings:
        r = [x - 1 for x in r]  # 0-index
        counts = [0] * n_categories
        for x in r:
            if 0 <= x < n_categories:
                counts[x] += 1
        n_i = k
        if n_i < 2: continue
        s2 = sum(c*c for c in counts)
        P_i = (s2 - n_i) / (n_i * (n_i - 1))
        P_i_list.append(P_i)
    if not P_i_list: return float('nan')
    P_bar = float(np.mean(P_i_list))

    # p_j 比例
    p_j = [0] * n_categories
    for r in ratings:
        r = [x - 1 for x in r]
        for x in r:
            if 0 <= x < n_categories:
                p_j[x] += 1
    total = sum(p_j)
    if total == 0: return float('nan')
    p_j = [c / total for c in p_j]
    P_e = sum(p*p for p in p_j)

    if abs(1 - P_e) < 1e-9: return float('nan')
    return (P_bar - P_e) / (1 - P_e)


def compute_fleiss_per_attr(nodule_to_ratings, min_readers=2):
    """
    对每个 attribute, 收集所有 nodule 的 reader ratings 列表, 计算 Fleiss K.
    只用至少 min_readers 个 reader 标了的 nodule.
    """
    results = {}
    for a in ATTR_NAMES:
        n_cat = ATTR_CLASSES[a]
        ratings_list = []
        for nid, readers in nodule_to_ratings.items():
            attr_ratings = []
            for rad, attrs in readers.items():
                if a in attrs:
                    attr_ratings.append(attrs[a])
            if len(attr_ratings) >= min_readers:
                ratings_list.append(attr_ratings)
        if not ratings_list:
            results[a] = {'K': float('nan'), 'n_items': 0}
            continue
        K = fleiss_kappa(ratings_list, n_cat)
        results[a] = {'K': K, 'n_items': len(ratings_list)}
    return results


def compute_agreement(ratings, n_categories):
    """原始一致率 (raw agreement): 对每个 item, 找众数占比"""
    if not ratings: return 0.0
    agree_count = 0
    for r in ratings:
        if not r: continue
        from collections import Counter
        c = Counter(r)
        top = c.most_common(1)[0][1]
        agree_count += top / len(r)
    return agree_count / len(ratings)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--xml_root', default=r'D:\datasets\LIDC_first\LIDC-IDRI',
                        help='LIDC 原始 XML 根目录')
    parser.add_argument('--ratings_json', default=None,
                        help='预抽取的 per-rater ratings JSON, 跳过 XML 扫描')
    parser.add_argument('--out_json', default=None, help='结果输出 JSON')
    parser.add_argument('--min_readers', type=int, default=2,
                        help='至少几个 reader 评过才计入 (默认 2, 论文用 4)')
    args = parser.parse_args()

    if args.ratings_json and os.path.exists(args.ratings_json):
        with open(args.ratings_json, encoding='utf-8') as f:
            nodule_to_ratings = json.load(f)
        print(f'从 {args.ratings_json} 加载 {len(nodule_to_ratings)} nodule', flush=True)
    else:
        if not os.path.isdir(args.xml_root):
            print(f'ERROR: {args.xml_root} 不存在, 请用 --ratings_json 传入预抽取数据')
            return
        nodule_to_ratings = build_ratings_from_xml(args.xml_root)

    results = compute_fleiss_per_attr(nodule_to_ratings, min_readers=args.min_readers)

    # 计算 raw agreement 一起输出
    agreement_results = {}
    for a in ATTR_NAMES:
        n_cat = ATTR_CLASSES[a]
        ratings_list = []
        for nid, readers in nodule_to_ratings.items():
            attr_ratings = []
            for rad, attrs in readers.items():
                if a in attrs:
                    attr_ratings.append(attrs[a])
            if len(attr_ratings) >= args.min_readers:
                ratings_list.append(attr_ratings)
        agr = compute_agreement(ratings_list, n_cat)
        agreement_results[a] = agr

    # 打印
    print()
    print('=' * 80)
    print(f'【Fleiss K (reader ≥ {args.min_readers})】')
    print('=' * 80)
    print(f'{"Attribute":<20} {"Fleiss K":<12} {"Raw Agree":<12} {"n_items":<10}')
    for a in ATTR_NAMES:
        r = results[a]
        K = r['K']
        K_str = f'{K:.4f}' if not np.isnan(K) else 'N/A'
        print(f'{a:<20} {K_str:<12} {agreement_results[a]:<12.4f} {r["n_items"]:<10}')

    # 输出 JSON
    if args.out_json:
        out = {
            'min_readers': args.min_readers,
            'per_attr': {
                a: {
                    'fleiss_K': float(results[a]['K']) if not np.isnan(results[a]['K']) else None,
                    'raw_agreement': float(agreement_results[a]),
                    'n_items': results[a]['n_items'],
                } for a in ATTR_NAMES
            }
        }
        with open(args.out_json, 'w', encoding='utf-8') as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f'\n结果: {args.out_json}')


if __name__ == '__main__':
    main()

"""9 征象的标注完整度统计 (用于论文 "各征象的四位医师标注完整占比、单医师标注占比")

对每个 unique nodule 跨 4 readingSession 配对:
  - 统计该 nodule 出现次数 (= 几个医师标了)
  - 9 个征象的标注医师数 (实际应一致, 但 LIDC 数据集可能不一致)
"""
import os
import re
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from itertools import combinations

XMLNS = '{http://www.nih.gov}'
DS_ROOT = r'D:\datasets\LIDC_first\LIDC-IDRI'
PATIENT_RE = re.compile(r'(LIDC-IDRI-\d+)')

# 9 个属性
ATTR_NAMES = ['malignancy', 'subtlety', 'internalStructure', 'calcification',
              'sphericity', 'margin', 'lobulation', 'spiculation', 'texture']


def is_lidc_ct_xml(xml_path):
    try:
        with open(xml_path, 'rb') as f:
            head = f.read(4096).decode('utf-8', errors='ignore')
        return 'LidcReadMessage' in head and 'IdriReadMessage' not in head
    except:
        return False


def parse_xml(xml_path):
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        session = root.find(XMLNS + 'readingSession')
        if session is None:
            return None
        rad_node = session.find(XMLNS + 'servicingRadiologistID')
        rad_id = rad_node.text if (rad_node is not None and rad_node.text is not None) else 'unknown'
        m = PATIENT_RE.search(xml_path)
        pid = m.group(1) if m else 'unknown'
        # 配对 key: (patient_id, round_x, round_y) — 忽略 sop (同 nodule 跨 3-slice 算 1 个)
        attr_to_nodules = {a: set() for a in ATTR_NAMES}
        for n in session.findall(XMLNS + 'unblindedReadNodule'):
            ch = n.find(XMLNS + 'characteristics')
            if ch is None:
                continue
            # 找 nodule 的 bbox (nodule-level, 跨所有 roi 取平均)
            xs_all, ys_all = [], []
            for roi in n.findall(XMLNS + 'roi'):
                edges = roi.findall('.//ns:edgeMap', {'ns': 'http://www.nih.gov'})
                for e in edges:
                    xs_all.append(int(e.find(XMLNS + 'xCoord').text))
                    ys_all.append(int(e.find(XMLNS + 'yCoord').text))
            if not xs_all:
                continue
            # 配对 key: (patient_id, round_x_20, round_y_20) — 容差 ±20 像素
            cx = (min(xs_all)+max(xs_all))/2
            cy = (min(ys_all)+max(ys_all))/2
            key = (pid, int(cx) // 20 * 20, int(cy) // 20 * 20)
            for a in ATTR_NAMES:
                node = ch.find(XMLNS + a)
                if node is not None and node.text is not None:
                    attr_to_nodules[a].add(key)
        return (rad_id, attr_to_nodules)
    except:
        return None


def main():
    # 收集所有 rad × attr × nodule_key
    attr_rad_to_nodules = {a: defaultdict(set) for a in ATTR_NAMES}
    # rad_to_pids
    rad_attrs = defaultdict(int)  # 统计每个 rad 出现次数
    skipped_cxr = 0
    total = 0

    for root, _, files in os.walk(DS_ROOT):
        for fn in files:
            if not fn.lower().endswith('.xml'):
                continue
            xml_path = os.path.join(root, fn)
            if not is_lidc_ct_xml(xml_path):
                skipped_cxr += 1
                continue
            total += 1
            res = parse_xml(xml_path)
            if res is None:
                continue
            rad_id, attr_to_nodules = res
            rad_attrs[rad_id] += 1
            for a in ATTR_NAMES:
                for k in attr_to_nodules[a]:
                    attr_rad_to_nodules[a][rad_id].add(k)

    # 1) 概览
    print('=' * 75)
    print('【概览】')
    print('=' * 75)
    print(f'扫到 LIDC CT XML 总数: {total}  (跳过 CXR: {skipped_cxr})')
    print(f'Unique radiologist ID: {len(rad_attrs)}')
    for r, c in sorted(rad_attrs.items()):
        print(f'  {r}: {c} 个 readingSession')

    # 2) 每个征象的 rad × nodule 矩阵
    print()
    print('=' * 75)
    print('【表 1】每个 radiologist 对每个征象的 unique nodule 标注数')
    print('=' * 75)
    print(f"{'属性':<18}", end='')
    for r in sorted(rad_attrs.keys()):
        print(f' {r:<15}', end='')
    print(f' {"4医师并集":<10}')
    for a in ATTR_NAMES:
        print(f'{a:<18}', end='')
        union_nodules = set()
        for r in sorted(rad_attrs.keys()):
            nn = len(attr_rad_to_nodules[a][r])
            print(f' {nn:<15}', end='')
            union_nodules |= attr_rad_to_nodules[a][r]
        print(f' {len(union_nodules):<10}')

    # 3) 关键: 每个征象, 多少 unique nodule 被 k 个医师标了
    print()
    print('=' * 75)
    print('【表 2】每个征象的"被 k 个医师标注"分布 (unique nodule 数)')
    print('=' * 75)
    # 对每个征象, 对每个 unique nodule, 数它被几个 rad 标了
    # 先建 attr_nodule -> set(rad)
    print(f"{'属性':<18} {'1 医师':<10} {'2 医师':<10} {'3 医师':<10} {'4 医师':<10} {'unique 总数':<10}")
    for a in ATTR_NAMES:
        # 统计
        nodule_to_rads = defaultdict(set)
        for r, nodules in attr_rad_to_nodules[a].items():
            for k in nodules:
                nodule_to_rads[k].add(r)
        c = Counter(len(r) for r in nodule_to_rads.values())
        total_u = len(nodule_to_rads)
        print(f'{a:<18} {c.get(1, 0):<10} {c.get(2, 0):<10} {c.get(3, 0):<10} {c.get(4, 0):<10} {total_u:<10}')

    # 4) 占比
    print()
    print('=' * 75)
    print('【表 3】每个征象的"标注完整度"占比 (基于 unique nodule 数)')
    print('=' * 75)
    print(f"{'属性':<18} {'1医师%':<10} {'2医师%':<10} {'3医师%':<10} {'4医师%':<10}")
    for a in ATTR_NAMES:
        nodule_to_rads = defaultdict(set)
        for r, nodules in attr_rad_to_nodules[a].items():
            for k in nodules:
                nodule_to_rads[k].add(r)
        c = Counter(len(r) for r in nodule_to_rads.values())
        total_u = max(len(nodule_to_rads), 1)
        p1 = c.get(1, 0) / total_u * 100
        p2 = c.get(2, 0) / total_u * 100
        p3 = c.get(3, 0) / total_u * 100
        p4 = c.get(4, 0) / total_u * 100
        print(f'{a:<18} {p1:<10.1f} {p2:<10.1f} {p3:<10.1f} {p4:<10.1f}')

    # 5) 论文 14,742 / 9,690 验证
    print()
    print('=' * 75)
    print('【表 4】论文数字验证')
    print('=' * 75)
    # 14,742 = 所有 rad 对 malignancy 标注的 nodule 总和 (不去重)
    total_14k = sum(len(s) for s in attr_rad_to_nodules['malignancy'].values())
    # 9,690 = malignancy 跨 rad 去重后被 ≥ 3 医师标注的 (因为 > 2/3 严格 = 至少 3 医师, 但 3 医师 2-1 不算, 所以 = 4 医师都标)
    nodule_to_rads_mal = defaultdict(set)
    for r, nodules in attr_rad_to_nodules['malignancy'].items():
        for k in nodules:
            nodule_to_rads_mal[k].add(r)
    n4 = sum(1 for rads in nodule_to_rads_mal.values() if len(rads) == 4)
    n_total = len(nodule_to_rads_mal)
    print(f'4 个医师对 malignancy 各自标注的 nodule 总和 (不去重): {total_14k}  (论文: 14,742)')
    print(f'4 医师都标注了 malignancy 的 unique nodule: {n4}  (论文严格 >2/3 共识: 9,690)')
    print(f'malignancy unique nodule 总数: {n_total}')


if __name__ == '__main__':
    main()

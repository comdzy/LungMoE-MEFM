import os
import xml.etree.ElementTree as ET

def extract_data_and_write_to_txt(xml_folder_path, output_folder):
    for root, dirs, files in os.walk(xml_folder_path):
        for filename in files:
            if filename.lower().endswith('.xml'):
               xml_file_path = os.path.join(root, filename)
               result = extract_data(xml_file_path)
               print(result)
               if result is not None:
                  for image_sop_uid, data in result.items():
                      txt_file_path = os.path.join(output_folder, f"{image_sop_uid}.txt")
                      write_to_txt(data, txt_file_path)

def extract_data(xml_file_path):
    # 解析XML文件
    tree = ET.parse(xml_file_path)
    root = tree.getroot()

    # 定义命名空间
    xmlns = '{http://www.nih.gov}'

    # 找到所有readingSession
    readingSession = root.findall(xmlns + 'readingSession')  # )#readingSession   characteristics unblindedReadNodule
    #print("解析文件：",xml_file_path)
    #print("reading_session:",reading_session)
    readingSession_len = len(readingSession)#判断共有几个readingSession，若没有readingSession 退出，若有readingSession
                                            # 对readingSession进行遍历
    if readingSession_len==0 :
        print(f"No reading session found in {xml_file_path}")
        return None
    # if reading_session is None:
    #     print(f"No reading session found in {xml_file_path}")
    #     return None
    result = {}

    for i in range(readingSession_len):#遍历所有的readingSession
        unblindedReadNodule = readingSession[i].findall(xmlns + 'unblindedReadNodule')
        unblindedReadNodule_len = len(unblindedReadNodule)
        print('第',i+1,'个readingSession','unblindedReadNodule_len',unblindedReadNodule_len)
        for j in range(unblindedReadNodule_len):
            characteristics = unblindedReadNodule[j].find(xmlns + 'characteristics')#找到第一个characteristics
            zcharacteristics = unblindedReadNodule[j].findall(xmlns + 'characteristics')#找到所有的characteristics
            print('第',j+1,'个unblindedReadNodule共有',len(zcharacteristics),'个characteristics')
            print('zcharacteristics',zcharacteristics)
            if characteristics:
                print('characteristics', characteristics)
                print("解析文件：", xml_file_path)
                malignancy=characteristics.find(xmlns + 'malignancy').text#得到characteristics下malignancy的值
                subtlety=characteristics.find(xmlns + 'subtlety').text
                internalStructure=characteristics.find(xmlns + 'internalStructure').text
                calcification=characteristics.find(xmlns + 'calcification').text
                sphericity=characteristics.find(xmlns + 'sphericity').text
                margin=characteristics.find(xmlns + 'margin').text
                lobulation=characteristics.find(xmlns + 'lobulation').text
                spiculation=characteristics.find(xmlns + 'spiculation').text
                texture=characteristics.find(xmlns + 'texture').text
                rois = unblindedReadNodule[j].findall(xmlns + 'roi') #得到characteristics下所有roi

                # 存储每个imageSOP_UID对应的roi信息和标签数值
                for roi in rois:
                    image_sop_uid = roi.find(xmlns + 'imageSOP_UID').text
                    # 获取边界框信息
                    edge_maps = roi.findall('.//ns:edgeMap', {'ns': 'http://www.nih.gov'})
                    x_coords = [int(edge.find(xmlns + 'xCoord').text) for edge in edge_maps]
                    y_coords = [int(edge.find(xmlns + 'yCoord').text) for edge in edge_maps]
                    left = min(x_coords)
                    top = min(y_coords)
                    right = max(x_coords)
                    bottom = max(y_coords)
                    width = right - left
                    height = bottom - top
                    center_x = (left + right) / 2
                    center_y = (top + bottom) / 2
                    # 存储边界框信息
                    bbox = {
                        'center': (center_x, center_y),
                        'width': width,
                        'height': height
                    }


                    # 存储imageSOP_UID和对应的roi信息和malignancy数值
                    if image_sop_uid in result:
                        result[image_sop_uid]['rois'].append(bbox)
                        result[image_sop_uid]['malignancy'].append(malignancy)
                        result[image_sop_uid]['subtlety'].append(subtlety)
                        result[image_sop_uid]['internalStructure'].append(internalStructure)
                        result[image_sop_uid]['calcification'].append(calcification)
                        result[image_sop_uid]['sphericity'].append(sphericity)
                        result[image_sop_uid]['margin'].append(margin)
                        result[image_sop_uid]['lobulation'].append(lobulation)
                        result[image_sop_uid]['spiculation'].append(spiculation)
                        result[image_sop_uid]['texture'].append(texture)
                    else:
                        result[image_sop_uid] = {'rois': [bbox], 
                                                 'malignancy': [malignancy], 
                                                 'subtlety': [subtlety],
                                                 'internalStructure': [internalStructure], 
                                                 'calcification': [calcification], 
                                                 'sphericity': [sphericity], 
                                                 'margin': [margin], 
                                                 'lobulation': [lobulation], 
                                                 'spiculation': [spiculation], 
                                                 'texture': [texture]}

        return result

def write_to_txt(data, txt_file_path):
    # 将数据写入txt文件
    with open(txt_file_path, 'w') as f:
        for i, roi_info in enumerate(data['rois']):
            f.write(f"ROI {i+1}:\n")
            f.write(f" Center: {roi_info['center']}")
            f.write(f" Width: {roi_info['width']}")
            f.write(f" Height: {roi_info['height']}")
            f.write(f" Malignancy: {data['malignancy'][i]}")
            f.write(f" Subtlety: {data['subtlety'][i]}")
            f.write(f" InternalStructure: {data['internalStructure'][i]}")
            f.write(f" Calcification: {data['calcification'][i]}")
            f.write(f" Sphericity: {data['sphericity'][i]}")
            f.write(f" Margin: {data['margin'][i]}")
            f.write(f" Lobulation: {data['lobulation'][i]}")
            f.write(f" Spiculation: {data['spiculation'][i]}")
            f.write(f" Texture: {data['texture'][i]}\n")
        
if __name__ == "__main__":
    # 提供包含XML文件的文件夹路径和保存txt文件的文件夹路径
    xml_folder_path = r'D:\CTData\lung\LIDC\LIDC-IDRI'
    output_folder = r'D:\CTData\lung\Dataset\txt'

    # 创建 TXT 保存文件夹
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    # 提取数据并将其写入到txt文件
    extract_data_and_write_to_txt(xml_folder_path, output_folder)








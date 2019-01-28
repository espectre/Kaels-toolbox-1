from __future__ import print_function
import json
import pprint


def load_categories(cat_path):
    '''
    load category file as a list
    '''
    with open(cat_path,'r') as f:
        categories = [x.strip() for x in f.readlines()]
    return categories


def _int_coors(coor_list):
    coor_res = list()
    for x in coor_list:
        coor_res.append(int(float(x)))
    return coor_res


def load_annotations(json_path):
    '''
    load coco official bounding-box annotations file
    '''
    raw = list()
    with open(json_path,'r') as f:
        raw_coco = json.load(f)
    print('=> Annotation info:')
    pprint.pprint(raw_coco['info'])
    categories = ['__background__']
    # print('[{}], {}'.format(0, categories[0]))
    for cat in raw_coco['categories']:
        categories.append(cat['name']) 
    print('=> {} categories loaded:'.format(len(categories)))
    count = [0 for i in range(len(categories))]
    bboxes = raw_coco['annotations']
    for box in bboxes:
        tmp = dict()
        tmp['ImageID'] = box['image_id']
        tmp['Source'] = 'coco_ann'
        tmp['LabelName'] = categories[int(box['category_id'])]
        count[int(box['category_id'])] += 1
        tmp['Confidence'] = 1
        assert len(box['bbox'])==4, "bounding box error: {}".format(box)
        tmp['XMin'], tmp['YMin'] = box['bbox'][:2]
        tmp['XMax'], tmp['YMax'] = tmp['XMin']+box['bbox'][2], tmp['YMin']+box['bbox'][3]
        tmp['XMin'],tmp['XMax'],tmp['YMin'],tmp['YMax'] = _int_coors([tmp['XMin'],tmp['XMax'],tmp['YMin'],tmp['YMax']])
        raw.append(tmp)
    print('=>', len(raw), 'bounding-boxes loaded.')
    for idx,cat in enumerate(categories):
        print('[{}], {}, {} boxes'.format(idx, cat, count[idx]))
    return raw


if __name__ == '__main__':
    load_annotations('/workspace/alpha/blademaster/data/juggernaut/juggdet/juggdet_0124/juggdet_1211_train_0124.json')

import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join
import xml.sax


root_path = getcwd()
anno_path = os.path.join(root_path, 'Annotations')
jpeg_path = os.path.join(root_path, 'JPEGImages')
img_path = os.path.join(root_path, 'ImageSets')

sets =['train', 'test']

table = ['table',
         'table occluded',
         'night table occluded',
         'table crop',
         'night table',
         'coffee table',
         'side table',
         'small table',
         'night table crop',
         'billiard table',
         'side table occluded',
         'coffee table occluded',
         'conference table',
         'billiard table crop',
         'console table',
         'coffee table crop',
         'billiard table occluded',
         'side table crop',
         'console table occluded',
         'end table',
         'television table',
         'card table',
         'console table crop',
         'small table occluded',
         'work table',
         'auxiliary table',
         'dining table',
         'table game occluded',
         'television table occluded',
         'dressing table',
         'end table occluded',
         'card table crop',
         'coffe table occluded',
         'glass table',
         'trolley table',
         'auxiliar table crop',
         'bedside table',
         'caffee table',
         'checkers table',
         'chess table',
         'cut table',
         'dressing table crop',
         'dressing table occluded',
         'instrument table',
         'instruments table',
         'judge table',
         'massage table',
         'secretary table',
         'set table',
         'set table stool',
         'side table',
         'small table crop',
         'tea table crop',
         'tv table']

classes = [table]


def convert(size, box):
        dw = 1./size[0]
        dh = 1./size[1]
        x = (box[0] + box[1])/2.0
        y = (box[2] + box[3])/2.0
        w = box[1] - box[0]
        h = box[3] - box[2]
        x = x*dw
        w = w*dw
        y = y*dh
        h = h*dh
        return (x,y,w,h)


#def convert_annotation(year, image_id):
#    in_file = open('VOCdevkit/VOC%s/Annotations/%s.xml'%(year, image_id))
#    out_file = open('VOCdevkit/VOC%s/labels/%s.txt'%(year, image_id), 'w')
#    tree=ET.parse(in_file)
#    root = tree.getroot()
#    size = root.find('size')
#    w = int(size.find('width').text)
#    h = int(size.find('height').text)
#
#    for obj in root.iter('object'):
#        difficult = obj.find('difficult').text
#        cls = obj.find('name').text
#        if cls not in classes or int(difficult) == 1:
#            continue
#        cls_id = classes.index(cls)
#        xmlbox = obj.find('bndbox')
#        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
#        bb = convert((w,h), b)
#        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')
def get_all_file_names(path):

   return  os.path.listdir(path)

# root_path: the root dir of the voc dataset
# filename: pass to a full .xml  name
def convert_annotation_to_label(root_path, filename):
#    anno_path = os.path.join(root_path, 'Annotations')
    root = ET.getroot(os.path.join(anno_path, filename))
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    label_path = os.path.join(root_path, 'label')
    if not os.path.exists(label_path):
        os.makedirs(label_path)

    label_file = open(os.path.join(label_path, filename.split('.')[0] + '.txt'), 'w')


    for obj in root.iter('object'):
        difficult = obj.find('difficult')
        classname = obj.find('name').text
        if classname not in classes[0] or int(difficult)==1:
            continue
        classname_id = classes.index[classname]
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text),
             float(xmlbox.find('xmax').text),
             float(xmlbox.find('ymin').text),
             float(xmlbox.find('ymax').text))
        bb = convert((w, h), b)
        label_file.write(str(classname_id) + " " + " ".join([str(a) for a in bb]) + '\n')
    label_file.close()


def get_train_test(root_path):
#   img_path = os.path.join(root_path, 'ImageSets')
    main_path = os.path.join(img_path, 'Main')
    filenames = get_all_file_names(main_path)

    for filename in filenames:
        #read ImageSet/Main to get img_id
        datafile = open(os.path.join(main_path, filename)).read().split()
        lines = datafile.readlines()
        # write train.txt and test.txt to root_path/train.txt  and
        # root_path/test.txt
        list_file = open(filename, 'w')
        for line in lines:
            list_file.write(os.path.join(jpeg_path, line + '.jpg') + '\n')
            convert_annotation_to_label(root_path, line + ".xml")
        list_file.close()


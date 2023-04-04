import glob
import os

import cv2

#src_path = '/data/smb/syh/gland_segmentation/CRAGV2/CRAG/train/'
#src_path = '/data/hdd1/by/House-Prices-Advanced-Regression-Techniques/data/Warwick QU Dataset (Released 2016_07_08)'

#dest_img = '/data/smb/syh/gland_segmentation/CRAGV2/Cropped/train/Images'
#dest_anno = '/data/smb/syh/gland_segmentation/CRAGV2/Cropped/train/Annotation'

#dest_img = '/data/smb/syh/gland_segmentation/Glas/Cropped/train/Images'
#dest_anno = '/data/smb/syh/gland_segmentation/Glas/Cropped/train/Annotation'
count = 0
import numpy as np



import transforms


trans = transforms.Resize(
    range=[0.5, 1.5],
    # size=(256, 256)
    size=(224, 224)
)


def crop_patches(img, crop_size):
    img_res = []

    h, w = img.shape[:2]
    for r_idx in range(0, h - int(crop_size / 2), int(crop_size / 2)):
        if r_idx + crop_size > h - 1:
            r_idx = h - crop_size - 1
        for c_idx in range(0, w - int(crop_size / 2), int(crop_size / 2)):
            if c_idx + crop_size > w - 1:
                c_idx = w - crop_size - 1

            imgpatch = img[r_idx:r_idx + crop_size, c_idx:c_idx + crop_size]

            assert imgpatch.shape[0] == crop_size
            assert imgpatch.shape[1] == crop_size

            img_res.append(imgpatch)

    return img_res

def overlay(img, mask):

    overlay = np.zeros(img.shape, img.dtype)


    overlay[mask > 0] = (0, 255, 0)

    alpha = 0.7
    beta = 1 - alpha
    return cv2.addWeighted(img, alpha, overlay, beta, 0.0)


def get_imgfilenames_crag(path):
    for i in glob.iglob(os.path.join(path, '**', '*.png'), recursive=True):
        if 'Overlay' in i:
            continue

        if 'Annotation' in i:
            continue

        yield i


def convert_to_segmap_crag(img_filename):
    return img_filename.replace('Images', 'Annotation')


def convert_to_segmap_glas(img_filename):
    return img_filename.replace('.bmp', '_anno.bmp')

def get_imgfilenames_glas_train(path):
    for i in glob.iglob(os.path.join(path, '**', '*.bmp'), recursive=True):



        if '_anno.bmp' in i:
            continue

        if 'test' in i :
            continue

        yield i

def get_imgfilenames_glas_test(path):
    for i in glob.iglob(os.path.join(path, '**', '*.bmp'), recursive=True):



        if '_anno.bmp' in i:
            continue

        if 'test'  not in i :
            continue

        yield i

# for i in glob.iglob(os.path.join(src_path, '**', '*.bmp'), recursive=True):
#     if 'Overlay' in i:
#         continue

#     if 'Annotation' in i:
#         continue

    #if 'test' in i:
    #    continue

    #if '_anno.bmp' in i:
    #    continue

    # segmap_filename = i.replace('Images', 'Annotation')
    #segmap_filename = i.replace('.bmp', '_anno.bmp')
    #print(segmap_filename)

    #print(i)
    #print(segmap_filename)
    #continue

#src_path = '/data/smb/syh/gland_segmentation/CRAGV2/CRAG/valid/'
#dest_img = '/data/smb/syh/gland_segmentation/CRAGV2/Cropped/valid/Images'
#dest_anno = '/data/smb/syh/gland_segmentation/CRAGV2/Cropped/valid/Annotation'



#src_path = '/data/smb/syh/gland_segmentation/CRAGV2/CRAG/train/'
#dest_img = '/data/smb/syh/gland_segmentation/CRAGV2/Cropped/train/Images'
#dest_anno = '/data/smb/syh/gland_segmentation/CRAGV2/Cropped/train/Annotation'

#src_path = '/data/hdd1/by/House-Prices-Advanced-Regression-Techniques/data/Warwick QU Dataset (Released 2016_07_08)'
#dest_img = '/data/smb/syh/gland_segmentation/Glas/Cropped/train/Images'
#dest_anno = '/data/smb/syh/gland_segmentation/Glas/Cropped/train/Annotation'


src_path = '/data/hdd1/by/House-Prices-Advanced-Regression-Techniques/data/Warwick QU Dataset (Released 2016_07_08)'
dest_img = '/data/smb/syh/gland_segmentation/Glas/Cropped/valid/Images'
dest_anno = '/data/smb/syh/gland_segmentation/Glas/Cropped/valid/Annotation'


glas_test_args = {
    'src_path' : '/data/hdd1/by/House-Prices-Advanced-Regression-Techniques/data/Warwick QU Dataset (Released 2016_07_08)',
    'dest_img': '/data/smb/syh/gland_segmentation/Glas/Cropped/valid/Images',
    'dest_anno': '/data/smb/syh/gland_segmentation/Glas/Cropped/valid/Annotation',
    'file_lists' : get_imgfilenames_glas_test,
    'converter' : convert_to_segmap_glas,
    'prefix' : 'test',
    'crop_size' : 384,
}

glas_train_args = {
    'src_path' : '/data/hdd1/by/House-Prices-Advanced-Regression-Techniques/data/Warwick QU Dataset (Released 2016_07_08)',
    'dest_img': '/data/smb/syh/gland_segmentation/Glas/Cropped/train/Images',
    'dest_anno': '/data/smb/syh/gland_segmentation/Glas/Cropped/train/Annotation',
    'file_lists' : get_imgfilenames_glas_train,
    'converter' : convert_to_segmap_glas,
    'prefix' : 'train',
    # 'crop_size' : 224,
    'crop_size' : 384,
}


crag_train_args = {
    'src_path' : '/data/smb/syh/gland_segmentation/CRAGV2/CRAG/train/',
    'dest_img': '/data/smb/syh/gland_segmentation/CRAGV2/Cropped/train/Images',
    'dest_anno': '/data/smb/syh/gland_segmentation/CRAGV2/Cropped/train/Annotation',
    'converter' : convert_to_segmap_crag,
    'file_lists' : get_imgfilenames_crag,
    'prefix' : 'train',
    'crop_size' : 512,
}

crag_test_args = {
    'src_path' : '/data/smb/syh/gland_segmentation/CRAGV2/CRAG/valid/',
    'dest_img': '/data/smb/syh/gland_segmentation/CRAGV2/Cropped/valid/Images',
    'dest_anno': '/data/smb/syh/gland_segmentation/CRAGV2/Cropped/valid/Annotation',
    'converter' : convert_to_segmap_crag,
    'file_lists' : get_imgfilenames_crag,
    'prefix' : 'test',
    'crop_size' : 512,
}

def run(args):
    src_path = args.get('src_path')
    dest_img = args.get('dest_img')
    dest_anno = args.get('dest_anno')
    file_lists = args.get('file_lists')
    prefix = args.get('prefix')
    crop_size = args.get('crop_size')
    converter = args.get('converter')
#for i in get_imgfilenames_crag(src_path):
#for i in get_imgfilenames_glas_train(src_path):
    count = 0
    for i in file_lists(src_path):

        #segmap_filename = convert_to_segmap_crag(i)
        print(i)
        segmap_filename = converter(i)

        img = cv2.imread(i, -1)
        segmap = cv2.imread(segmap_filename, -1)

        #imgpatches = crop_patches(img, 512)
        #segpatches = crop_patches(segmap, 512)

        imgpatches = crop_patches(img, crop_size)
        segpatches = crop_patches(segmap, crop_size)
        #print(img.shape, np.unique(segmap))

        for imgpatch, segpatch in zip(imgpatches, segpatches):

                #cv2.imwrite(os.path.join('tmp', '{}_{}.jpg'.format(prefix, count)), overlay(imgpatch, segpatch))

                #print(os.path.join(dest_img, 'test_{}.jpg'.format(count)))
                #print(os.path.join(dest_anno, 'test_{}.png'.format(count)))
                cv2.imwrite(os.path.join(dest_img, '{}_{}.jpg'.format(prefix, count)), imgpatch)
                cv2.imwrite(os.path.join(dest_anno, '{}_{}.png'.format(prefix, count)), segpatch)



                count += 1
                #if count == 50:
                #    import sys; sys.exit()
                #print(r_idx, c_idx)
    print(count)
run(crag_train_args)
run(crag_test_args)
run(glas_train_args)
run(glas_test_args)

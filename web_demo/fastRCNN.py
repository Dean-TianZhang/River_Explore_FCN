import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import scipy.misc as scimc
import caffe, os, sys, cv2
import argparse
import time
from PIL import Image

CLASSES = ('__background__',
           'water_gauge')

NETS = {'vgg16': ('VGG16',
                  'VGG16_faster_rcnn_final.caffemodel'),
        'zf': ('ZF',
                  'ZF_water_gauge_final.caffemodel')}

# input ans output path
# in_path = './water_gauge_data/'
# out_path = './water_gauge'

def vis_detections(image_name, im, class_name, dets, thresh=0.5):
    out_detect = './water_gauge/gauge_detect/'
    out_crop = './water_gauge/gauge_crop/'

    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        os.system('cp %s %s' % (
                  image_name,
                  os.path.join(out_detect, 'no_objs_'+image_name)))
        return

    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    filename = []
    widthList = []
    heightList = []
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='blue', linewidth=5)
            )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=32, color='white')
        
        w = bbox[2]-bbox[0]
        h = bbox[3]-bbox[1]
        r = round(h/w,1)
        crop_name = image_name.split('/')[-1][:-4]+"_"+str(int(w))+'x'+str(int(h))+'-'+str(r)+'_'+str(i)+".jpg"
        filepath = os.path.join(out_crop, crop_name)
        cropped = Image.fromarray(im).crop(bbox).save(filepath)
        filename.append(filepath)
        widthList.append(w)
        heightList.append(h)
    # Find the max width detected gauge for visualization
    loc = widthList.index(np.max(widthList))
    # ax.set_title(('{} detections with '
    #              'p({} | box) >= {:.1f}').format(class_name, class_name,
    #                                              thresh),
    #              fontsize=14)
    plt.axis('off')
    plt.tight_layout()
#     plt.draw()
    detection = os.path.join(out_detect, image_name.split('/')[-1])
    plt.savefig(detection,bbox_inches='tight', pad_inches=0)
    return detection,filename[loc],widthList[loc]/heightList[loc]
    plt.close(fig)

def fig2data (fig):
    # draw the renderer
    fig.canvas.draw ( )
 
    # Get the RGBA buffer from the figure
    w,h = fig.canvas.get_width_height()
    buf = np.fromstring ( fig.canvas.tostring_argb(), dtype=np.uint8 )
    buf.shape = ( w, h,4 )
 
    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll ( buf, 3, axis = 2 )
    return buf

def demo(net, image_name):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    im = cv2.imread(image_name)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im)
    timer.toc()
    print ('Detect ' + image_name + ' took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    # Visualize detections for each class
    CONF_THRESH = 0.8
    NMS_THRESH = 0.3
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        detect,cropfile,w_h_ratio = vis_detections(image_name, im, cls, dets, thresh=CONF_THRESH)
    return detect,cropfile,w_h_ratio

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='demo_net', help='Network to use [zf]',
                        choices=NETS.keys(), default='zf')

    args = parser.parse_args()

    return args

def fastrcnn(im_name):
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals
    args = parse_args()
    prototxt = "./model/faster_rcnn_test.pt"
    caffemodel = "./model/ZF_water_gauge_final.caffemodel"

    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\nDid you run ./data/script/'
                       'fetch_faster_rcnn_models.sh?').format(caffemodel))

    if args.cpu_mode:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
        cfg.GPU_ID = args.gpu_id
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    print '\n\nLoaded network {:s}'.format(caffemodel)

    # Warmup on a dummy image
    im = 128 * np.ones((300, 500, 3), dtype=np.uint8)
    for i in xrange(2):
        _, _= im_detect(net, im)
    
    # "im_name" is the path link to target image
    starttime = time.time()
    detection, cropFile,ratio = demo(net, im_name)
    endtime = time.time()

    # plt.show()
    return detection,cropFile,ratio,'%.3f' % (endtime - starttime)

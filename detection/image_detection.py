#download model
#https://github.com/tensorflow/models
#https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md
#https://www.youtube.com/watch?v=Rgpfk6eYxJA

import tensorflow as tf
import numpy as np
import cv2

object_ = ['','person','bicycle','car','motorcycle','airplane','bus','train','truck','boat'
            ,'traffic light','fire hydrant','stop sign','parking meter','bench','bird','dog','cat','horse','sheep'
            ,'cow','elephant','bear','zebra','giraffe','backpack','umbrella','handbag','tie','suitcase'
            ,'frisbee','skis','snowboard','sports ball','kite','baseball bat','baseball glove','skateboard','surfboard'
           ,'tennis racket','bottle','wine glass','cup','fork','knife','spoon','bowl','banana','apple','sandwich'
           ,'orange','broccoli','carrot','hot dog','pizza','donut','cake','chair','couch','potted plant','bed','dining table'
           ,'toilet''tv','laptop','mouse','remote','keyboard','cell phone','microwave','oven','toaster','sink','refrigerator','book','clock'
           ,'vase','scissors','teddy bear','hair drier','toothbrush']

graph_path = './models/ssd_mobilenet_v2.pb'

acc = 0.6

detection_graph = tf.Graph()

with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(graph_path, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
            sess = tf.Session(graph=detection_graph)

image_tensor      = detection_graph.get_tensor_by_name('image_tensor:0')
detection_boxes   = detection_graph.get_tensor_by_name('detection_boxes:0')
detection_scores  = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
num_detections    = detection_graph.get_tensor_by_name('num_detections:0')

image_path     = './images/cat001.jpg'
image          = cv2.imread(image_path)
#print(image)
image_expanded = np.expand_dims(image, axis=0) #เพิ่มมิติ
#print(image_expanded )
#exit()
(boxes, scores, classes, num) = sess.run([detection_boxes, detection_scores, detection_classes, num_detections],feed_dict={image_tensor: image_expanded})

#print(scores)
#print('=========================')
scores  = np.squeeze(scores) #คำสั่ง .squeeze() จะกำจัดมิติที่มีค่าเดียวทิ้ง
#print(scores)
boxes   = np.squeeze(boxes)
classes = np.squeeze(classes).astype(np.int32)
#print(classes)

height , width , d = image.shape
for i in range(0,len(scores)):
    if scores[i] > acc:
        ymin, xmin, ymax, xmax = boxes[i]
        x1 = int(xmin * width)
        x2 = int(xmax * width)
        y1 = int(ymin * height)
        y2 = int(ymax * height)
        cv2.rectangle(image, (x1,y1), (x2, y2), (255, 255, 0), 2)
        cv2.putText(image, object_[classes[i]] ,(x1,y1),cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),2,cv2.LINE_AA)
        print(object_[classes[i]])

cv2.imshow('image',image)
cv2.waitKey(0)

cv2.destroyAllWindows()

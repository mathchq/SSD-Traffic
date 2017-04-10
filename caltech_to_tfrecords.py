# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Convert a dataset to TFRecords format, which can be easily integrated into
a TensorFlow pipeline.
"""

import tensorflow as tf

import os
import sys

import json

from datasets.caltech_common import LABELS
from datasets.dataset_utils import int64_feature, float_feature, bytes_feature

def main(_):

    print('Dataset directory: ./datasets')
    print('Output directory: ./datasets')
    print('Output name: caltech')

    tf_filename = './datasets/caltech.tfrecord'
    if tf.gfile.Exists(tf_filename):
        print('Dataset files already exist. Exiting without re-creating them.')
        return

    image_path = os.path.join('./datasets', 'JPEGImages/')
    annotations_path = os.path.join('./datasets', 'Annotations/')
    print('image path: ', image_path)
    print('annotations_path: ', annotations_path)
    images = sorted(os.listdir(image_path))

    annotations_file = annotations_path+'annotations.json'
    annotations_text = open(annotations_file)
    annotations_json = json.load(annotations_text)
    annotations_frames = annotations_json['set01']['V000']['frames']

    input_height = 640
    input_width = 480
    input_depth = 3

    bboxes = []
    labels = []
    labels_text = []
    difficult = []
    truncated = []
    for i, frame in enumerate(sorted(map(int, list(annotations_frames.keys())))):
        sys.stdout.write('\r>> Annotating image %d/%d' % (i + 1, len(list(annotations_frames.keys()))))
        bboxes_f = []
        labels_f = []
        labels_text_f = []
        difficult_f = []
        truncated_f = []
        object_dicts_list = annotations_frames[str(frame)]
        for object_dict in object_dicts_list:
            if object_dict['lbl'] == 'person':
                label_f = 'person'
                labels_f.append(int(LABELS[label_f][0]))
                labels_text_f.append(label_f.encode('ascii'))

                difficult_f.append(0)
                truncated_f.append(0)

                pos = object_dict['pos']
                bboxes_f.append((float(pos[1]) / input_height,
                       float(float(pos[0])) / input_width,
                       float(pos[1]+pos[3]) / input_height,
                       float(pos[0]+pos[2]) / input_width
                       ))
        bboxes.append(bboxes_f)
        labels.append(labels_f)
        labels_text.append(labels_text_f)
        difficult.append(difficult_f)
        truncated.append(truncated_f)

    with tf.python_io.TFRecordWriter(tf_filename) as tfrecord_writer:
        for i, imagename in enumerate(images):
            sys.stdout.write('\r>> Converting image %d/%d' % (i + 1, len(images)))
            sys.stdout.flush()

            image_file = image_path+imagename
            image_data = tf.gfile.FastGFile(image_file, 'r').read()

            xmin = []
            ymin = []
            xmax = []
            ymax = []

            for b in bboxes[i]:
                [l.append(point) for l, point in zip([xmin, ymin, xmax, ymax], b)]
            image_format = b'JPEG'
            example = tf.train.Example(features=tf.train.Features(feature={
                'image/height': int64_feature(input_height),
                'image/width': int64_feature(input_width),
                'image/channels': int64_feature(input_depth),
                'image/shape': int64_feature([input_height, input_width, input_depth]),
                'image/object/bbox/xmin': float_feature(xmin),
                'image/object/bbox/xmax': float_feature(xmax),
                'image/object/bbox/ymin': float_feature(ymin),
                'image/object/bbox/ymax': float_feature(ymax),
                'image/object/bbox/label': int64_feature(labels[i]),
                'image/object/bbox/label_text': bytes_feature(labels_text[i]),
                'image/object/bbox/difficult': int64_feature(difficult[i]),
                'image/object/bbox/truncated': int64_feature(truncated[i]),
                'image/format': bytes_feature(image_format),
                'image/encoded': bytes_feature(image_data)}))
            tfrecord_writer.write(example.SerializeToString())

    print('\nFinished converting the Pascal VOC dataset!')





if __name__ == '__main__':
    tf.app.run()



from absl import app
from absl import flags

import tensorflow.compat.v2 as tf
from google.protobuf import text_format
from object_detection import exporter_lib_v2
from object_detection.protos import pipeline_pb2

tf.enable_v2_behavior()


FLAGS = flags.FLAGS

flags.DEFINE_string('input_type', 'image_tensor', 'Type of input node. Can be '
                    'one of [`image_tensor`, `encoded_image_string_tensor`, '
                    '`tf_example`, `float_image_tensor`, '
                    '`image_and_boxes_tensor`]')
flags.DEFINE_string('pipeline_config_path', ' ',
                    'Path to a pipeline_pb2.TrainEvalPipelineConfig config '
                    'file.')
flags.DEFINE_string('trained_checkpoint_dir', ' ',
                    'Path to trained checkpoint directory')
flags.DEFINE_string('output_directory', ' ', 'Path to write outputs.')
flags.DEFINE_string('config_override', '',
                    'pipeline_pb2.TrainEvalPipelineConfig '
                    'text proto to override pipeline_config_path.')
flags.DEFINE_boolean('use_side_inputs', False,
                     'If True, uses side inputs as well as image inputs.')
flags.DEFINE_string('side_input_shapes', '',
                    'If use_side_inputs is True, this explicitly sets '
                    'the shape of the side input tensors to a fixed size. The '
                    'dimensions are to be provided as a comma-separated list '
                    'of integers. A value of -1 can be used for unknown '
                    'dimensions. A `/` denotes a break, starting the shape of '
                    'the next side input tensor. This flag is required if '
                    'using side inputs.')
flags.DEFINE_string('side_input_types', '',
                    'If use_side_inputs is True, this explicitly sets '
                    'the type of the side input tensors. The '
                    'dimensions are to be provided as a comma-separated list '
                    'of types, each of `string`, `integer`, or `float`. '
                    'This flag is required if using side inputs.')
flags.DEFINE_string('side_input_names', '',
                    'If use_side_inputs is True, this explicitly sets '
                    'the names of the side input tensors required by the model '
                    'assuming the names will be a comma-separated list of '
                    'strings. This flag is required if using side inputs.')

flags.mark_flag_as_required('pipeline_config_path')
flags.mark_flag_as_required('trained_checkpoint_dir')
flags.mark_flag_as_required('output_directory')


def main(_):
  pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
  with tf.io.gfile.GFile(FLAGS.pipeline_config_path, 'r') as f:
    text_format.Merge(f.read(), pipeline_config)
  text_format.Merge(FLAGS.config_override, pipeline_config)
  exporter_lib_v2.export_inference_graph(
      FLAGS.input_type, pipeline_config, FLAGS.trained_checkpoint_dir,
      FLAGS.output_directory, FLAGS.use_side_inputs, FLAGS.side_input_shapes,
      FLAGS.side_input_types, FLAGS.side_input_names)


if __name__ == '__main__':
  app.run(main)

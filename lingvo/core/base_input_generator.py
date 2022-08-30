# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Input generators."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
from six.moves import range
import tensorflow as tf

from tensorflow.contrib.tpu.python.tpu import tpu_function
from tensorflow.python.framework import function
from tensorflow.python.ops import io_ops
from lingvo.core import base_layer
from lingvo.core import input_generator_helper as ig_helper
from lingvo.core import py_utils
from lingvo.core import tokenizers
from lingvo.core.ops import py_x_ops


_TFDATA_DATASET_CTORS = {
    'tfrecord': tf.data.TFRecordDataset,
    'textline': tf.data.TextLineDataset,
    'csv': tf.contrib.data.CsvDataset,
}


def _LookupTfDataDatasetCtor(factory_type):
  """Looks up a tf.data dataset constructor by factory_type.

  Args:
    factory_type: One of the well-known string type names. It can also be a
      class, in which case, it will be instantiated and used as the constructor.
      In this case, the class should implement __call__.

  Returns:
    A callable that takes a list of filenames (see tf.data.TFRecordDataset
    for the convention).
  Raises:
    ValueError: If an unknown type.
  """
  if isinstance(factory_type, type):
    factory = factory_type()
    if not callable(factory):
      raise ValueError('Dataset factory %r is not callable' % factory_type)
    return factory

  if factory_type == 'tfrecord':
    return tf.data.TFRecordDataset
  elif factory_type == 'textline':
    return tf.data.TextLineDataset
  elif factory_type == 'csv':
    return tf.contrib.data.CsvDataset

  raise ValueError('Unsupported tf.data dataset factory %s (known %r)' %
                   (factory_type, _TFDATA_DATASET_CTORS))


class BaseInputGenerator(base_layer.BaseLayer):
  """The base input generator."""

  @classmethod
  def Params(cls):
    """Defaults params for input generators."""
    p = super(BaseInputGenerator, cls).Params()
    p.name = 'input'
    p.Define('batch_size', 0, 'Batch size.')
    p.Define(
        'num_samples', 0,
        'If non-zero, the dataset contains these many samples. '
        'For test/eval dataset, if we want the test/evel job evaluate '
        'the whole dataset, this param must be set precisely. Otherwise, '
        'this param is optional.')

    # TPU related infeed tuning.
    p.Define('use_per_host_infeed', False,
             'Whether run infeed op on each host.')
    p.Define('tpu_infeed_parallism', 1,
             'Uses these many python threads to drive infeed concurrently.')
    return p

  @base_layer.initializer
  def __init__(self, params):
    super(BaseInputGenerator, self).__init__(params)
    self._made_tpu_infeed = False
    # parameter to tell the bprop one hot for all the files.
    self._bprop_onehot = tf.constant([1], dtype=tf.float32)
    # Each entry is a regular expression specifying the set of variables
    # to bprop per data source.
    self._bprop_variable_filters = ['']

  def CommonInputOpArgs(self):
    """Common input params."""
    return {}

  def GetBpropVariableFilters(self):
    return self._bprop_variable_filters

  def GetInputSourceOneHot(self):
    """Get the current bprop type of the input generator batch."""
    return self._bprop_onehot

  def InputBatchSize(self):
    """Returns the batch size for the current step."""
    p = self.params
    cluster = self.cluster

    # If use_per_host_infeed, each input op is only responsible
    # for generating a subset of the whole batch.
    batch_per_input = p.batch_size * cluster.num_splits_per_client
    if p.use_per_host_infeed and cluster.num_tpu_hosts > 0:
      tf.logging.info('batch_size %d cluster.num_tpu_hosts %d', batch_per_input,
                      cluster.num_tpu_hosts)
      batch_per_input //= cluster.num_tpu_hosts
    tf.logging.info('batch_per_input: %d', batch_per_input)
    return batch_per_input

  def InputBatch(self):
    """The current input batch, not preprocessed.

    This is meant to be overridden by subclasses, but not called directly.
    Callers should use `GetPreprocessedInputBatch()`.

    Returns:
      A `.NestedMap` of input tensors. Each tensor's dim-0 must be the same
      and denotes the batch dimension.
    """
    raise NotImplementedError('Abstract method')

  def GetPreprocessedInputBatch(self):
    return self.PreprocessInputBatch(self.InputBatch())

  def PreprocessInputBatch(self, batch):
    # Can be overridden by subclasses.
    return batch

  def CreateTpuFeeds(self):
    """Creates the TPU infeed queue from preprocessed batch."""
    p = self.params
    cluster = self.cluster
    num_tpu_hosts = cluster.num_tpu_hosts
    assert num_tpu_hosts > 0, ('num_tpu_hosts: %d' % num_tpu_hosts)
    num_infeed_hosts = num_tpu_hosts if p.use_per_host_infeed else 1

    with py_utils.outside_all_rewrites():
      assert py_utils.use_tpu()
      assert not self._made_tpu_infeed

      shards = tpu_function.get_tpu_context(
      ).number_of_shards // num_infeed_hosts
      input_ops_list = []
      queues = []
      first_batch = None
      for task_id in range(num_infeed_hosts):
        host_device = '/task:{}/device:CPU:0'.format(task_id)
        with tf.device(host_device):
          batch = self.GetPreprocessedInputBatch()
          if first_batch is None:
            first_batch = batch
          flat_batch = batch.FlattenItems()

          shapes, types = [], []
          for k, x in flat_batch:
            assert x.shape.is_fully_defined(), (
                'Shape must be fully defined: %s: %s' % (k, x))
            # TODO(cwhipkey): if it's a string (or other type not supported on
            # TPU), drop it from feeding and on the other end add in an op that
            # fails if used.
            shapes.append(x.shape)
            types.append(x.dtype)
          q = tf.contrib.tpu.InfeedQueue(tuple_types=types, tuple_shapes=shapes)
          queues.append(q)
          assert shards is not None
          q.set_number_of_shards(shards)

          if p.use_per_host_infeed:

            # TODO(ylc/zhifengc): Add this to a policy module and test it.
            def _tpu_ordinal_function(shard_index_in_host):
              device_assignment = py_utils.GetTpuDeviceAssignment()
              if device_assignment:
                # We put both enqueue/dequeue ops at core 0 in each replica.
                replica = device_assignment.lookup_replicas(
                    task_id, 0)[shard_index_in_host]  # pylint: disable=cell-var-from-loop
                return device_assignment.tpu_ordinal(replica=replica)
              else:
                return shard_index_in_host

            input_ops = q.split_inputs_and_generate_enqueue_ops(
                [v for _, v in flat_batch],
                placement_function=lambda x: host_device,  # pylint: disable=cell-var-from-loop
                tpu_ordinal_function=_tpu_ordinal_function)
          else:
            input_ops = q.split_inputs_and_generate_enqueue_ops(
                [v for _, v in flat_batch],
                device_assignment=py_utils.GetTpuDeviceAssignment())

          input_ops_list += input_ops
      tf.logging.info('input_ops_list %s', input_ops_list)
      tpu_infeed_op = tf.group(*input_ops_list)
    self._made_tpu_infeed = True
    # Let trainer.py use multiple threads to drive the infeed op.
    for _ in range(p.tpu_infeed_parallism):
      tf.add_to_collection(py_utils.ENQUEUE_OPS, tpu_infeed_op)

    with tf.device(tf.contrib.tpu.core(0)):
      tensors = queues[0].generate_dequeue_op()
    return first_batch.Pack(tensors)

  def SplitInputBatch(self, num_splits):
    """Splits the current InputBatch into num_splits ways.

    Args:
      num_splits: The number of splits.

    Returns:
      A list of `.NestedMap`. Each `.NestedMap` represents the input
      tensors in one split.
    """
    assert num_splits >= 1

    batch = self.InputBatch()
    if num_splits == 1:
      # Special case. No split is needed.
      return [batch]

    assert not py_utils.use_tpu()
    field_split = ig_helper.SplitTensors(batch.Flatten(), num_splits)
    num_fields = len(field_split)
    ret = []
    for j in range(num_splits):
      split_flatten = [field_split[i][j] for i in range(num_fields)]
      split = batch.Pack(split_flatten)
      ret += [split]
    return ret


class BaseInputGeneratorFromFiles(BaseInputGenerator):
  """Base class for input generators that reads from files."""

  @classmethod
  def Params(cls):
    """Defaults params for input generators."""
    p = super(BaseInputGeneratorFromFiles, cls).Params()
    p.Define(
        # TODO(tilarids): Consider renaming this param to make it clear it may
        # be a set of file patterns with weights and filters and not just a
        # simple string pattern.
        'file_pattern',
        '',
        'A single file pattern string, a list of <file_pattern, weight> pairs'
        'or a list of  <file_pattern, weight, bprop_variable_filter> tuples.'
        'In the later 2 cases, probablistic samples are from the inputs '
        'proportional to their weights. Typically, values are binary '
        'protocol buffers containing train/eval samples. Keys are not used.')
    p.Define('file_random_seed', 301,
             'Random seed for shuffling the input data.')
    p.Define(
        'file_buffer_size', 10000,
        'How many records are buffered for random shuffling. This param '
        'affects how much RAM a train/test job needs. E.g., if an average '
        'record is about 500KB, the buffer needs 5GB ram.')
    p.Define('file_parallelism', 16, 'How many files to read concurrently.')
    p.Define(
        'flush_every_n', 0, 'If non-zero, flushes all batches buffered '
        'so far every these many records are yielded.')
    p.Define('num_batcher_threads', 1, 'Number of threads to use for input '
             'record batcher.')
    p.Define(
        'use_within_batch_mixing', False, 'Whether to mix records from '
        'different input sources within batch or across batches (the '
        'default option). This option only takes effect when file_pattern'
        ' is a list of file patterns with weights.')
    return p

  @base_layer.initializer
  def __init__(self, params):
    super(BaseInputGeneratorFromFiles, self).__init__(params)
    if self.params.use_per_host_infeed and self.params.file_random_seed != 0:
      raise ValueError('file_random_seed needs to be 0 when '
                       'use_per_host_infeed == True.')

  def CommonInputOpArgs(self):
    """Common input params."""
    p = self.params
    args = super(BaseInputGeneratorFromFiles, self).CommonInputOpArgs()
    args.update({
        'file_random_seed': p.file_random_seed,
        'file_buffer_size': p.file_buffer_size,
        'file_parallelism': p.file_parallelism,
        'flush_every_n': p.flush_every_n,
        'num_threads': p.num_batcher_threads,
    })
    args.update(self._InputOpBucketingArgs())
    return args

  def _InputOpBucketingArgs(self):
    return {
        'bucket_upper_bound': [1000000000],
        'bucket_batch_limit': [self.InputBatchSize()],
    }

  def _DataSourceFromFilePattern(self, file_pattern, input_source_weights=None):
    """Read and return input batch from a string file_pattern.

    Args:
      file_pattern: A string file pattern.
      input_source_weights: A list of float input source weights to control
        input example mix in the batch. The records will be sampled from inputs
        proportionally to these weights. Defaults to None which should be
        treated as an empty list.

    Returns:
      A tf.Tensor or `.NestedMap` of tf.Tensor
    """
    raise NotImplementedError()

  def _BuildWithinBatchMixingDataSource(self):
    """Read and return input batch from a p.file_pattern list.

    `p.file_pattern` should be a list of (file_pattern, weight) pairs. Examples
    in the batch will be mixed together from different file_pattern source
    proportionally to the weights.

    Returns:
      A tf.Tensor or `.NestedMap` of tf.Tensor same as
      `self._DataSourceFromFilePattern()`.

    Raises:
      ValueError: If unknown token type.
    """
    p = self.params
    if not isinstance(p.file_pattern, list):
      raise ValueError('Expected a list, got %s' % (p.file_pattern,))
    if max(map(len, p.file_pattern)) >= 3:
      # Within batch mixing doesn't work with backprop filters, i.e. when
      # file_pattern param contains a list of
      # <file_pattern, weight, [bprop_variable_filter]> tuples.
      raise ValueError('Expected a list of pairs, got %s' % (p.file_pattern,))

    file_patterns, weights = zip(*p.file_pattern)
    self._bprop_variable_filters = [''] * len(file_patterns)
    for file_pattern in file_patterns:
      if ',' in file_pattern:
        raise ValueError('Can not use commas in file_pattern when within-batch '
                         'mixing is used. file_pattern: %s' % (file_pattern,))

    # Do not set self._bprop_onehot as it shouldn't be used later on.
    try:
      return self._DataSourceFromFilePattern(','.join(file_patterns), weights)
    except TypeError as e:
      raise NotImplementedError(
          'The function may not be fully implemented. Have you added ' +
          'input_source_weights as params? Original exception: %s' % e)

  def _BuildCrossBatchMixingDataSource(self):
    """Read and return input batch from a p.file_pattern list.

    `p.file_pattern` should be a list of (file_pattern, weight,
    optional_bprop_filter) tuples. Every batch returned will be filled from one
    source only and batches will be mixed proportionally to the weights.
    Additionally some backprop filters may be applied for different input
    sources.

    Returns:
      A tf.Tensor or `.NestedMap` of tf.Tensor same as
      `self._DataSourceFromFilePattern()`.

    Raises:
      ValueError: If unknown token type.
    """
    p = self.params
    input_file_pattern = p.file_pattern

    def _MakeDataSourceFromFilePatternFunc(file_pattern):
      # It's important to invoke self._DataSourceFromFilePattern() inside the
      # lambda to make sure that the record is drawn from data source
      # only if it will be used.
      return lambda: self._DataSourceFromFilePattern(file_pattern)

    inputs = []
    weights = []
    self._bprop_variable_filters = []
    for input_entry in input_file_pattern:
      file_pattern, weight = input_entry[:2]
      inputs.append(_MakeDataSourceFromFilePatternFunc(file_pattern))
      weights.append(weight)
      bprop_variable_filter = input_entry[2] if len(input_entry) > 2 else ''
      self._bprop_variable_filters.append(bprop_variable_filter)
    data_source, selected_bprop = py_utils.MixByWeight(inputs, weights)
    self._bprop_onehot = selected_bprop
    return data_source

  def _BuildDataSource(self):
    """Read and return input batch from `p.file_pattern`.

    `p.file_pattern` may be a string file_pattern or a
    list of (file_pattern, weight, [bprop_variable_filter]) tuples.
    bprop_variable_filter is optional. When bprop_variable_filter is used,
    batches will always contain the examples from the same source. Otherwise,
    examples from different sources may be mixed together.

    Returns:
      A tf.Tensor or `.NestedMap` of tf.Tensor same as
      `self._DataSourceFromFilePattern()`.

    Raises:
      ValueError: If unknown token type.
    """
    p = self.params
    input_file_pattern = p.file_pattern
    if isinstance(input_file_pattern, six.string_types):
      return self._DataSourceFromFilePattern(input_file_pattern)
    elif isinstance(input_file_pattern, list):
      if p.use_within_batch_mixing:
        return self._BuildWithinBatchMixingDataSource()
      else:
        # Otherwise fall back to MixByWeight-based approach.
        return self._BuildCrossBatchMixingDataSource()
    else:
      raise ValueError()


class BaseSequenceInputGenerator(BaseInputGeneratorFromFiles):
  """The basic sequence input generator."""

  @classmethod
  def Params(cls):
    """Defaults params for sequence input generators."""
    p = super(BaseSequenceInputGenerator, cls).Params()
    p.Delete('batch_size')
    # How input should be bucketized.
    p.Define(
        'bucket_upper_bound', [2560], 'Bucketing scheme. Required to be'
        'a sorted list of integers. Examples that are longer than all bucket'
        'upper bounds are skipped.')
    p.Define(
        'bucket_batch_limit', [8], 'For each bucket, desired batch size. '
        'Must be the same length as bucket_upper_bound.')
    p.Define('source_max_length', None,
             'The maximum length of the source sequence.')
    p.Define('target_max_length', 300,
             'The maximum length of the target sequence.')
    p.Define('pad_to_max_seq_length', False,
             'If True, input tensors will be padded to max_length.')
    p.Define('tokenizer', tokenizers.AsciiTokenizer.Params(),
             'Tokenizer params.')
    p.Define(
        'tokenizer_dict', {},
        'If multiple tokenizers are required, they can be accessed through '
        'this dict via a key.')
    return p

  @base_layer.initializer
  def __init__(self, params):
    super(BaseSequenceInputGenerator, self).__init__(params)

    p = self.params
    self._input_batch_size = None

    if p.tokenizer:
      assert 'default' not in p.tokenizer_dict
      p.tokenizer_dict['default'] = p.tokenizer

    self.tokenizer_dict = {}
    for k, p in six.iteritems(p.tokenizer_dict):
      if p:
        name = '_tokenizer_' + k
        self.CreateChild(name, p)
        self.tokenizer_dict[k] = self.children[name]
      else:
        self.tokenizer_dict[k] = None

    self.tokenizer = self.tokenizer_dict['default']

  @property  # Adjust batch size according to the cluster spec.
  def scaled_bucket_batch_limit(self):
    p = self.params
    if not hasattr(self, '_scaled_bucket_batch_limit'):
      cluster = self.cluster
      self._scaled_bucket_batch_limit = [
          b * cluster.num_splits_per_client for b in p.bucket_batch_limit
      ]
      if p.use_per_host_infeed and cluster.num_tpu_hosts > 0:
        self._scaled_bucket_batch_limit = [
            x // cluster.num_tpu_hosts for x in self._scaled_bucket_batch_limit
        ]
    return self._scaled_bucket_batch_limit

  def InputBatchSize(self):
    if self._input_batch_size is None:
      raise ValueError('No input batch size is defined.')
    return self._input_batch_size

  def _InputOpBucketingArgs(self):
    p = self.params
    bucket_batch_limit = self.scaled_bucket_batch_limit
    tf.logging.info('bucket_batch_limit %r', bucket_batch_limit)
    return {
        'bucket_upper_bound': p.bucket_upper_bound,
        'bucket_batch_limit': bucket_batch_limit,
    }

  def StringsToIds(self,
                   strs,
                   is_source=False,
                   external_max_length=None,
                   external_append_eos=None,
                   key=None,
                   languages=None):
    """Tokenize strs into vocab ids.

    Args:
      strs: A vector of strings.
      is_source: A bool to indicate whether to use `source_max_length` to pad
        'strs'.
      external_max_length: An int providing the max_length for strs.
      external_append_eos: Bool or None. If None, will be ignored and
        `params.append_eos` will be used. If bool, will determine if an eos
        symbol will be added to tokens.
      key: A string key in case the model has multiple tokenizers.
      languages: A vector of str with the same length as `strs`.

    Returns:
      A tuple (ids, labels, paddings) with the same shape [batch, maxlen].

      - ids[i, j] is the input token id of i-th sample for j-th step.
      - labels[i, j] is the target token id of i-th sample for j-th step.
      - paddings[i, j] is 1 iff i-th sample's j-th step is padded.

    Raises:
      ValueError: If unknown token type.
    """
    p = self.params

    if external_max_length is not None:
      maxlen = external_max_length
    elif is_source:
      maxlen = p.source_max_length
    else:
      maxlen = p.target_max_length

    key = key or 'default'
    return self.tokenizer_dict[key].StringsToIds(
        strs, maxlen, external_append_eos, languages=languages)

  def IdsToStrings(self, ids, lens, key=None):
    """Converts ids back to strings.

    Args:
      ids: A matrix of shape [batch, seqlen]. ids[i, :] is the i-th sample's
        ids.
      lens: A vector of shape [batch]. lens[i] is the sequence length of the
        i-th sample. Only the first lens[i] tokens in ids[i, :] are valid tokens
          for the i-th sequence.
      key: A string key in case the model has multiple tokenizers.

    Returns:
      sequences - A vector of shape [batch]. The converted string sequence.

    Raises:
      ValueError: If unknown token type.
    """
    key = key or 'default'
    return self.tokenizer_dict[key].IdsToStrings(ids, lens)


class BaseTinyDatasetInput(BaseInputGenerator):
  """Input generator for tiny dataset which are stored in tf checkpoint.

      | Input batch (b: batch size, h: height, w: width, d: depth):
      |   raw: Samples. [b, h, w, d].
      |   data: Preprocessed samples. [b, h, w, d].
      |   label: Labels. [b].
      |   weight: [b]. weight[i] is 1.0 if i-th sample is considered to
      |     be a real example. Otherwise, weight[i] is 0.0.
  """

  @classmethod
  def Params(cls):
    """Defaults params."""
    p = super(BaseTinyDatasetInput, cls).Params()
    p.Define('ckpt', None, 'A TensorFlow checkpoint.')
    p.Define('data', 'x_train', 'The tensor name in the ckpt.')
    p.Define('data_dtype', tf.uint8, 'The tensor dtype in the ckpt.')
    p.Define(
        'data_shape', (0, 0, 0), 'A tuple of ints. E.g., a tiny image '
        'has the shape (height, weight, depth).')
    p.Define('label', 'y_train', 'The tensor name in the ckpt.')
    p.Define('label_dtype', tf.uint8, 'The tensor dtype in the ckpt.')
    p.Define('repeat', True, 'If true, goes through the dataset repeatedly.')
    p.use_per_host_infeed = True
    return p

  def InputBatch(self):
    p = self.params

    @function.Defun()
    def ReadData():
      x, y = io_ops.restore_v2(p.ckpt, [p.data, p.label], [''] * 2,
                               [p.data_dtype, p.label_dtype])
      # Always convert to float32.
      return tf.to_float(x), tf.to_float(y)

    # Loads data and label into memory and keep it around.
    data, label = py_x_ops.cached_call(f=ReadData, T=[tf.float32, tf.float32])
    b, shape = self.InputBatchSize(), list(p.data_shape)
    data = tf.reshape(data, [-1] + shape)
    label = tf.reshape(label, [-1])
    label = py_utils.HasShape(label, [tf.shape(data)[0]])
    sample_ids = py_x_ops.random_permutation_sequence(
        num=p.num_samples,
        batch=b,
        repeat=p.repeat,
        seed=p.random_seed if p.random_seed else 0)
    n = tf.shape(sample_ids)[0]
    raw = py_utils.PadOrTrimTo(tf.gather(data, sample_ids), [b] + shape)
    ret = py_utils.NestedMap(
        raw=raw,
        data=self._Preprocess(raw),
        label=py_utils.PadOrTrimTo(tf.gather(label, sample_ids), [b]),
        weight=py_utils.PadOrTrimTo(tf.ones([n], dtype=tf.float32), [b]))
    if not py_utils.use_tpu():
      ret['sample_ids'] = sample_ids
    return ret

  def _Preprocess(self, raw):
    return raw


class BaseExampleInputGenerator(BaseInputGenerator):
  """Base class for input generators that read Feature protos via tf.data."""

  @classmethod
  def Params(cls):
    p = super(BaseExampleInputGenerator, cls).Params()
    p.Define('input_files', None, 'Delimited glob of input files.')
    p.Define(
        'file_type', None,
        'The file type. Can be one of "tfrecord" or a class that implements '
        '__call__ and takes a list of filenames (note that other file types '
        'are not valid for reading examples).')
    p.Define('randomize_order', True, 'Whether to randomize the order.')
    p.Define('parallel_readers', 1, 'Number of parallel reader threads.')
    p.Define('num_examples', -1, 'Number of examples (-1 for unlimited).')
    p.Define('randomize_shuffle_size', 500,
             'Size of the random shuffle buffer.')
    return p

  def __init__(self, params):
    super(BaseExampleInputGenerator, self).__init__(params)
    p = params
    assert p.input_files, (
        'input_files is required for a tf.data example input generator')
    assert p.file_type, (
        'file_type is required for a tf.data example input generator')

  def GetFeatureSpec(self):
    """Subclasses must implement and return a feature spec.

    Returns:
      NestedMap of features compatible with tf.parse_example. Default
      implementation returns an empty dict.
    """
    return {}

  def PostProcessBatch(self, input_batch):
    """Post processes an input batch.

    By default, just returns the batch unchanged. This happens as part of the
    parallel reader threads and can therefore be more efficient for performing
    expensive operations vs doing work as the result of InputBatch().

    Args:
      input_batch: Input batch NestedMap.

    Returns:
      Altered batch NestedMap.
    """
    return input_batch

  def InputBatch(self):
    p = self.params

    def ParseAndProcess(*cols):
      """Parses a Tensorflow example into features."""
      # Assume either one or two column input. If one, then the record is
      # assumed to be that column. If 2, then it is assumed to be a KV store
      # and the record is the second column.
      assert len(cols) in [
          1, 2
      ], ('BaseExampleInputGenerator supports one or two column input')
      record = cols[-1]
      feature_spec = self.GetFeatureSpec()
      features = py_utils.NestedMap(tf.parse_example(record, feature_spec))
      return self.PostProcessBatch(features)

    dataset_factory = _LookupTfDataDatasetCtor(p.file_type)
    dataset = (
        tf.data.Dataset.list_files(
            p.input_files, shuffle=bool(p.randomize_order)).apply(
                tf.contrib.data.parallel_interleave(
                    dataset_factory,
                    cycle_length=p.parallel_readers,
                    sloppy=p.randomize_order)))

    if p.randomize_order:
      dataset = dataset.shuffle(p.randomize_shuffle_size)
    dataset = dataset.take(p.num_examples)
    dataset = dataset.batch(p.batch_size, drop_remainder=True)
    dataset = dataset.map(
        ParseAndProcess, num_parallel_calls=p.parallel_readers)
    dataset = dataset.repeat()
    iterator = dataset.make_one_shot_iterator()
    input_batch = iterator.get_next()
    return input_batch

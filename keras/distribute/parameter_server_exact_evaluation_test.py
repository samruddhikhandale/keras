# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests for evaluation using Keras model and ParameterServerStrategy."""
import itertools
import re
import time

import numpy as np
import tensorflow.compat.v2 as tf
from absl.testing import parameterized
from tensorflow.python.platform import tf_logging as logging

import keras
from keras.metrics import base_metric
from keras.testing_infra import test_utils

# isort: off
from tensorflow.python.distribute import (
    multi_worker_test_base,
)
from tensorflow.python.distribute.cluster_resolver import (
    SimpleClusterResolver,
)


def standardize(name):
    # metric name format: alphabetic strings connected by underscores,
    # potentially followed by an underscore then a number.
    # e.g. "auc_1" or "mean_absolute_error"
    if re.fullmatch(r"[a-zA-Z]+(_[a-zA-Z]+)*(_[0-9]+)?", name):
        if name[-1].isnumeric():
            return "_".join(name.split("_")[:-1])
        return name


def match_standardized(names, target_name):
    for name in names:
        if standardize(name) == target_name:
            return name
    return None


def _aggregate_results(coordinator_metrics, results):
    for result in results:
        for metric in coordinator_metrics:
            matched_key = match_standardized(result.keys(), metric.name)
            metric_result = result[matched_key]
            assert len(metric_result) == len(metric.weights)
            for weight, val in zip(metric.weights, metric_result):
                weight.assign_add(val)
    return coordinator_metrics


@test_utils.run_v2_only
class ExactEvaluationTest(tf.test.TestCase, parameterized.TestCase):
    def setUp(self):
        super(ExactEvaluationTest, self).setUp()
        self._cluster = multi_worker_test_base.create_multi_process_cluster(
            num_workers=5, num_ps=1, rpc_layer="grpc"
        )
        self._cluster_def = (
            self._cluster.cluster_resolver.cluster_spec().as_dict()
        )
        cluster_resolver = SimpleClusterResolver(
            tf.train.ClusterSpec(self._cluster_def), rpc_layer="grpc"
        )

        self.strategy = tf.distribute.experimental.ParameterServerStrategy(
            cluster_resolver
        )
        self.cluster_coord = (
            tf.distribute.experimental.coordinator.ClusterCoordinator(
                self.strategy
            )
        )

    def tearDown(self):
        super(ExactEvaluationTest, self).tearDown()
        self._cluster.stop()
        self._cluster = None

    def testDistributedMetrics(self):
        coordinator_metrics = [
            keras.metrics.AUC(),
            keras.metrics.MeanAbsoluteError(),
        ]

        def dataset_fn():
            y_true = np.concatenate((np.zeros(512), np.ones(512)))
            y_pred = np.concatenate(
                (np.linspace(0, 1, 512), np.linspace(0, 1, 512))
            )
            return tf.data.Dataset.from_tensor_slices((y_true, y_pred)).batch(1)

        @tf.function
        def eval_shard_fn(total_shard, shard_id, worker_dataset):
            worker_metrics = []
            for coord_metric in coordinator_metrics:
                worker_metrics.append(
                    base_metric.clone_metric(coord_metric, is_local=True)
                )

            dataset_shard = worker_dataset.shard(total_shard, shard_id)

            for value in dataset_shard:
                for worker_metric in worker_metrics:
                    worker_metric.update_state(*value)

            return {metric.name: metric.weights for metric in worker_metrics}

        per_worker_dataset = self.cluster_coord.create_per_worker_dataset(
            dataset_fn()
        )
        # Trigger dataset creation on workers without creating an iterator
        built_dataset = per_worker_dataset.build()

        # needs to be a tf.constant so it doesn't get re-traced each time
        # needs to be int64 because that's what Dataset.shard expects
        total_shards = tf.constant(100, dtype=tf.int64)

        result_remote_values = []
        logging.info("Scheduling eval closures")
        for i in tf.range(total_shards):
            result_remote_values.append(
                self.cluster_coord.schedule(
                    eval_shard_fn, args=(total_shards, i, built_dataset)
                )
            )

        logging.info("Killing 2 workers")
        self._cluster.kill_task("worker", 0)
        self._cluster.kill_task("worker", 1)
        time.sleep(1)
        self._cluster.start_task("worker", 0)
        self._cluster.start_task("worker", 1)

        self.cluster_coord.join()
        results = [r.fetch() for r in result_remote_values]
        coordinator_metrics = _aggregate_results(coordinator_metrics, results)

        expected_results = {"auc": 0.5, "mean_absolute_error": 0.5}
        for metric in coordinator_metrics:
            self.assertAlmostEqual(
                metric.result().numpy(), expected_results[metric.name], places=5
            )

    def testModelAddMetricErrors(self):
        class MyModel(keras.Model):
            def call(self, x):
                self.add_metric(
                    tf.cast(x >= 0, tf.float32),
                    aggregation="sum",
                    name="num_positive",
                )
                return tf.cast(tf.add(x, 1), tf.float32)

        dataset = tf.data.Dataset.zip(
            (tf.data.Dataset.range(-5, 5), tf.data.Dataset.range(-4, 6))
        ).batch(1)
        with self.strategy.scope():
            model = MyModel()
            model.compile(
                metrics=[keras.metrics.Accuracy()], loss="binary_crossentropy"
            )

        # run a single train step to compile metrics
        model.fit(dataset, steps_per_epoch=1)
        with self.assertRaises(ValueError):
            model.evaluate(dataset, exact_evaluation="auto", return_dict=True)

    @parameterized.parameters(itertools.product([True, False], [True, False]))
    def testDistributedModelEvaluation(self, eval_in_model_fit, use_auto):

        # Define dataset by batch size, number of shards, and batches per shard
        batch_size = 16
        num_shards = 32
        batches_per_shard = 4
        num_examples = batch_size * num_shards * batches_per_shard

        # Input dataset x: just the sequence of numbers up to the dataset size
        # Input dataset y: defined such that each shard has index equal to the
        # number of y_i's == True in that shard
        expected_acc = sum(range(num_shards)) / num_examples

        # The predictions y_pred from this dummy model are fixed to True. This
        # way we can control the expected accuracy by just modifying y.
        class MyModel(keras.Model):
            def __call__(self, x, training=False):
                return tf.cast(x >= 0, tf.float32)

        def dataset_fn():
            x = np.arange(num_examples)

            def make_batch_with_n_true(n):
                return np.concatenate((np.ones(n), np.zeros(batch_size - n)))

            y = np.zeros(num_examples)
            batch_ixs = np.arange(num_examples // batch_size)
            for shard_ix in range(num_shards):
                num_correct = shard_ix
                # Dataset.shard uses mod sharding, so each shard consists of the
                # batches whose index mod (num_shards) = shard_ix
                batch_ixs_for_shard = np.where(
                    np.mod(batch_ixs, num_shards) == shard_ix
                )[0]
                for batch_ix in batch_ixs_for_shard:
                    # Select the individual data elements for this batch
                    batch_range = range(
                        batch_ix * batch_size, (batch_ix + 1) * batch_size
                    )
                    num_for_batch = min(num_correct, batch_size)
                    y[batch_range] = make_batch_with_n_true(num_for_batch)
                    num_correct -= num_for_batch

            dataset = tf.data.Dataset.from_tensor_slices((x, y))

            dataset = dataset.batch(batch_size)
            return dataset

        def build_metric():
            return keras.metrics.Accuracy()

        logging.info("Local evaluation (exact)")
        model = MyModel()
        model.compile(metrics=[build_metric()])
        ground_truth_evaluation = model.evaluate(dataset_fn())
        logging.info(
            "Result local evaluation (exact): %s", ground_truth_evaluation
        )
        self.assertAlmostEqual(ground_truth_evaluation[1], expected_acc)

        logging.info("Distributed evaluation (exact)")
        with self.strategy.scope():
            model = MyModel()
            model.compile(metrics=[build_metric()], loss="binary_crossentropy")

        dataset = dataset_fn()

        if use_auto:
            num_shards = "auto"
        else:
            num_shards = 5 * model.distribute_strategy._extended._num_workers
        expected_results = {"accuracy": expected_acc}

        eval_results = {}
        if eval_in_model_fit:
            history = model.fit(
                dataset,
                steps_per_epoch=1,
                validation_data=dataset,
                exact_evaluation=num_shards,
            )
            logging.info(
                "History: params (%r), history (%r)",
                history.params,
                history.history,
            )
            eval_results = {
                metric.split("val_")[1]: val[-1]
                for metric, val in history.history.items()
                if metric.startswith("val_")
            }
        else:
            # run a single train step to compile metrics
            model.fit(dataset, steps_per_epoch=1)
            eval_results = model.evaluate(
                dataset, exact_evaluation=num_shards, return_dict=True
            )
            eval_results = {
                metric: val.numpy() for metric, val in eval_results.items()
            }
        for metric, val in eval_results.items():
            if "loss" not in metric:
                self.assertIn(metric, expected_results)
                self.assertAlmostEqual(val, expected_results[metric], places=5)


if __name__ == "__main__":
    tf.__internal__.distribute.multi_process_runner.test_main()

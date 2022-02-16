import shutil
import gin
import logging
import os
import numpy as np
from absl import app, flags
from train import Trainer
import matplotlib.pyplot as plt
from input_pipeline import datasets
from models.model import rnn
from evaluation.eval import Evaluator
from models.tcn_model import model_tcn
from input_pipeline.make_tfrecords import create_tfrecords
from main import setup

FLAGS = flags.FLAGS
flags.DEFINE_integer("start", "2", "Specify the smallest fc")
flags.DEFINE_integer("end", "20", "Specify the biggest fc")
flags.DEFINE_integer("step", "2", "Specify the step")
flags.DEFINE_integer(
    "num", "5", "Number of runs that should be used to average the accuracy of one specification (defautl: 5)"
)
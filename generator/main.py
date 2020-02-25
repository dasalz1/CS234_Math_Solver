"""Generates question-answer pairs using DeepMind's Mathematics Dataset."""
from __future__ import absolute_import
from __future__ import print_function

import collections
import time

import six
from absl import app
from absl import flags
from absl import logging
from mathematics_dataset.modules import modules

import util

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "filter",
    "",
    "restrict to modules containing this string (ex. `algebra__polynomial_roots_composed).`",
)
flags.DEFINE_integer("num_examples", 128, "Num of examples to generate.")
flags.DEFINE_bool(
    "print_results", True, "Should print question-answer pairs (or not)."
)


def generate_problems(filter, difficulty, num_examples):
    """Generate question-answer pairs using the DeepMind Mathematics Dataset.

        Keyword arguments:
        filter -- only use modules that contain this keyword
        difficulty -- float between 0.0 and 1.0 corresponding to the entropy used in generating constants\
            for each problem type
        num_examples -- number of problems to generate for each module
        """
    problems = collections.defaultdict(lambda: [])
    initial_modules = modules.train(util._make_entropy_fn(difficulty, 1))
    filtered_modules = util._filter_and_flatten(filter, initial_modules)
    for module_name, module in six.iteritems(filtered_modules):
        # These magic print constants make the header bold.
        if FLAGS.print_results:
            print("\033[1m{}\033[0m".format(module_name))
        num_dropped = 0
        for _ in range(num_examples):
            problem, extra_dropped = util.sample_from_module(module)
            num_dropped += extra_dropped
            if FLAGS.print_results:
                print(f"Module name: {module_name}")
            problems[module_name].append(problem)
        if num_dropped > 0:
            if FLAGS.print_results:
                logging.warning("Dropped %d examples", num_dropped)
    return problems


def main(unused_argv):
    start_time = time.time()
    module_problems = generate_problems("algebra", 0.5, 10)
    end_time = time.time()
    for module_name, problems in module_problems.items():
        if FLAGS.print_results:
            print(f"\033[1m\nModule: {module_name}\033[0m")
        for problem in problems:
            if FLAGS.print_results:
                print(
                    f"\033[1mQuestion\033[0m: {problem.question}\n\033[1mAnswer\033[0m: {problem.answer}"
                )
    print(
        f"Generated {FLAGS.num_examples} question-answer pairs in each of {len(module_problems)} modules in {end_time - start_time} seconds.\nThat corresponds to {FLAGS.num_examples * len(module_problems) / (end_time - start_time)} problems per second."
    )


if __name__ == "__main__":
    app.run(main)

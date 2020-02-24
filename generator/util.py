from mathematics_dataset import generate_settings
import collections
import six

def _make_entropy_fn(level, num_levels):
  """This returns a function that returns a subrange of entropy.

  E.g., if level=1 (medium) and num_levels=3, then the returned function will
  map the range [x, x + y] to [x + y/3, x + 2y/3].=

  Args:
    level: Integer in range [0, num_levels - 1].
    num_levels: Number of difficulty levels.

  Returns:
    Function to restrict entropy range.
  """
  lower = level / num_levels
  upper = (level + 1) / num_levels
  def modify_entropy(range_):
    assert len(range_) == 2
    length = range_[1] - range_[0]
    return (range_[0] + lower * length, range_[0] + upper * length)
  return modify_entropy

def _filter_and_flatten(filter, modules_):
  """Returns flattened dict, filtered according to FLAGS."""
  flat = collections.OrderedDict()

  def add(submodules, prefix=None):
    for key, module_or_function in six.iteritems(submodules):
      full_name = prefix + '__' + key if prefix is not None else key
      if isinstance(module_or_function, dict):
        add(module_or_function, full_name)
      else:
        if filter not in full_name:
          continue
        flat[full_name] = module_or_function

  add(modules_)

  # Make sure list of modules are in deterministic order. This is important when
  # generating across multiple machines.
  flat = collections.OrderedDict(
      [(key, flat[key]) for key in sorted(six.iterkeys(flat))])

  return flat

def sample_from_module(module, show_dropped = False):
  """Samples a problem, ignoring samples with overly long questions / answers.

  Args:
    module: Callable returning a `Problem`.

  Returns:
    Pair `(problem, num_dropped)`, where `problem` is an instance of `Problem`
    and `num_dropped` is an integer >= 0 indicating the number of samples that
    were dropped.
  """
  num_dropped = 0
  while True:
    problem = module()
    question = str(problem.question)
    if len(question) > generate_settings.MAX_QUESTION_LENGTH:
      num_dropped += 1
      if show_dropped:
        logging.warning('Dropping question: %s', question)
      continue
    answer = str(problem.answer)
    if len(answer) > generate_settings.MAX_ANSWER_LENGTH:
      num_dropped += 1
      if show_dropped:
        logging.warning('Dropping question with answer: %s', answer)
      continue
    return problem, num_dropped


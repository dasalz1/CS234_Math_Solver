import collections
import six

import mathematics_dataset.generate_settings as generate_settings
import generator.util as util
from MathConstants import subcategories, full_categories


def _filter_and_flatten(categories, modules_):
    """Returns flattened dict, filtered according to FLAGS."""
    flat = collections.OrderedDict()
    sample_categories = []
    for category in categories:
        
        if category in sample_categories: continue
        if category in full_categories:
            for subc in subcategories[category]:
                sample_categories.append(category + '_' + subc)
        else:
            sample_categories.append(category)

    def add(submodules, prefix=None):
        for key, module_or_function in six.iteritems(submodules):
            full_name = prefix + '_' + key if prefix is not None else key
            if isinstance(module_or_function, dict):
                add(module_or_function, full_name)
            else:
                if full_name not in sample_categories:
                    continue
                flat[full_name] = module_or_function

    add(modules_)
    flat = collections.OrderedDict([(key, flat[key]) for key in sorted(six.iterkeys(flat))])

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
            continue
        answer = str(problem.answer)
        if len(answer) > generate_settings.MAX_ANSWER_LENGTH:
            num_dropped += 1
            continue
        return problem, num_dropped

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
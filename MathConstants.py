full_categories = ['algebra',
    'arithmetic',
    'calculus',
    'comparison',
    'measurement',
    'numbers',
    'polynomials',
    'probability']

algebra_subcategories = ['polynomial_roots',
            'linear_1d',
            'linear_2d',
            'sequence_nth_term',
            'sequence_next_term']


arithmetic_subcategories = ['add_or_sub',
            'add_sub_multiple',
            'add_or_sub_in_base',
            'mul',
            'mul_div_multiple',
            'div',
            'mixed',
            'nearest_integer_root','simplify_surd']

calculus_subcategories = ['differentiate']

comparison_subcategories = ['pair',
            'kth_biggest',
            'closest',
            'sort',]

measurement_categories = ['conversion', 'time']

numbers_categories = ['gcd',
            'lcm',
            'div_remainder',
            'is_prime',
            'is_factor',
            'round_number',
            'place_value',
            'list_prime_factors', ]

polynomials_categories = ['coefficient_named',
            'evaluate',
            'add',
            'expand',
            'collect',
            'compose',
            'simplify_power']

probability_categories = ['swr_p_sequence',
            'swr_p_level_set']

subcategories = {'algebra': algebra_subcategories, 'arithmetic': arithmetic_subcategories, 
				'calculus': calculus_subcategories, 'comparison': comparison_subcategories, 
				'measurement': measurement_categories,
				'numbers': numbers_categories, 'polynomials': polynomials_categories, 
				'probability': probability_categories}

all_subcategories = [subc for c in list(subcategories.values()) for subc in c]


def get_sub_categories(category):
	return subcategories[category]

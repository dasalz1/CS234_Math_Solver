full_categories = ['algebra',
    'arithmetic',
    'calculus',
    'comparison',
    'measurement',
    'numbers',
    'polynomials',
    'probability']

algebra_subcategories = ['polynomial_roots',
            'polynomial_roots_composed',
            'linear_1d',
            'linear_1d_composed',
            'linear_2d',
            'linear_2d_composed',
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

calculus_subcategories = ['differentiate_composed',  'differentiate']

comparison_subcategories = ['pair',
            'pair_composed',
            'kth_biggest',
            'kth_biggest_composed',
            'closest',
            'closest_composed',
            'sort',
            'sort_composed']

measurement_categories = ['conversion', 'time']

numbers_categories = ['gcd',
            'gcd_composed'
            'lcm',
            'lcm_composed',
            'div_remainder',
            'div_remainder_composed'
            'is_prime',
            'is_prime_composed',
            'is_factor',
            'is_factor_composed',
            'round_number',
            'round_number_composed',
            'place_value',
            'place_value_composed',
            'list_prime_factors', 
            'list_prime_factors_composed']

polynomials_categories = ['coefficient_named',
            'evaluate',
            'evaluate_composed',
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

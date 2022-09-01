OUTPUT_DATA_SPEC = {
    'stats': {
        'avg': {
            'groupings': ['month'],
            'columns': 'ALL'
        },
        'columns': ['month', 'price_index', 'gdp_index', 'gdp_growth', 'unemployment', 'average_workers',
                    'families_median_wealth', 'families_wealth', 'families_commuting', 'families_savings',
                    'families_helped', 'amount_subsidised', 'firms_wealth', 'firms_profit',
                    'gini_index', 'average_utility', 'pct_zero_consumption', 'rent_default', 'inflation', 'average_qli',
                    'house_vacancy', 'house_price', 'house_rent', 'affordable', 'p_delinquent', 'equally', 'locally',
                    'fpm', 'bank']
    },
    'families': {
        'avg': {
            'groupings': ['month', 'mun_id'],
            'columns': ['house_price', 'house_rent',
                        'total_wage', 'savings', 'num_members']
        },
        'columns': ['month', 'id', 'mun_id',
                    'house_price', 'house_rent',
                    'total_wage', 'savings', 'num_members']
    },
    'banks': {
        'avg': {
            'groupings': ['month'],
            'columns': 'ALL'
        },
        'columns': ['month', 'balance', 'deposits',
                    'active_loans', 'mortgage_rate', 'p_delinquent_loans',
                    'mean_loan_age', 'min_loan', 'max_loan', 'mean_loan']
    },
    'houses': {
        'avg': {
            'groupings': ['month', 'mun_id'],
            'columns': ['price', 'on_market']
        },
        'columns': ['month', 'id', 'x', 'y', 'size', 'price', 'rent', 'quality', 'qli',
                    'on_market', 'family_id', 'region_id', 'mun_id']
    },
    'firms': {
        'avg': {
            'groupings': ['month', 'firm_id'],
            'columns': ['total_balance$', 'number_employees',
                        'stocks', 'amount_produced', 'price', 'amount_sold',
                        'revenue', 'profit', 'wages_paid']
        },
        'columns':  ['month', 'firm_id', 'region_id', 'mun_id',
                     'long', 'lat', 'total_balance$', 'number_employees',
                     'stocks', 'amount_produced', 'price', 'amount_sold',
                     'revenue', 'profit', 'wages_paid']
    },
    'construction': {
        'avg': {
            'groupings': ['month', 'firm_id'],
            'columns': ['total_balance$', 'number_employees',
                        'stocks', 'amount_produced', 'price', 'amount_sold',
                        'revenue', 'profit', 'wages_paid']
        },
        'columns':  ['month', 'firm_id', 'region_id', 'mun_id',
                     'long', 'lat', 'total_balance$', 'number_employees',
                     'stocks', 'amount_produced', 'price', 'amount_sold',
                     'revenue', 'profit', 'wages_paid']
    },
    'regional': {
        'avg': {
            'groupings': ['month', 'mun_id'],
            'columns': 'ALL'
        },
        'columns': ['month', 'mun_id', 'commuting', 'pop', 'gdp_region',
                    'regional_gini', 'regional_house_values', 'regional_unemployment',
                    'qli_index', 'gdp_percapita', 'treasure', 'equally', 'locally', 'fpm',
                    'licenses']
    }
}

files = ['stats', 'regional', 'time', 'firms', 'banks', 'houses', 'agents', 'families', 'grave', 'construction']

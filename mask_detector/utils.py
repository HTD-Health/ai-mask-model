from sklearn.model_selection import train_test_split


# TODO: resolve lack of usage of train_ratio
def train_val_test_split(dataset, train_ratio, validate_ratio, test_ratio):
    # TODO: add docstrings
    remaining, test = train_test_split(dataset, test_size=test_ratio)

    remaining_ratio = 1 - test_ratio
    validate_ratio_adjusted = validate_ratio / remaining_ratio

    train, validate = train_test_split(
        remaining, test_size=validate_ratio_adjusted)

    return train, validate, test

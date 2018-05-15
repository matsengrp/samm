import numpy as np

from sklearn.model_selection import KFold
from sklearn.model_selection import GroupKFold

def split(num_obs, metadata, tuning_sample_ratio, number_folds=1, validation_column=None, val_column_idx=None):
    """
    @param num_obs: number of observations
    @param feat_generator: submotif feature generator
    @param metadata: metadata to include variables to perform validation on
    @param tuning_sample_ratio: ratio of data to place in validation set (only for when there is one fold)
    @param number_folds: number of folds for k-fold CV
    @param validation_column: variable to perform validation on (if None then sample randomly)
    @param val_column_idx: the category we want to force to the validation set (this is mostly for debugging)

    @return training and validation indices
    """
    assert not (tuning_sample_ratio > 0 and number_folds > 1)
    if number_folds == 1:
        train_idx, val_idx = split_train_val(num_obs, metadata, tuning_sample_ratio, validation_column=validation_column, val_column_idx=val_column_idx)
        return [(train_idx, val_idx)]
    else:
        fold_indices = split_kfolds(num_obs, metadata, number_folds, validation_column=validation_column)
        return fold_indices

def split_kfolds(num_obs, meta_data, n_splits, validation_column=None):
    """
    Split into k folds, by group if validation_column is specified
    @return list of tuples, each tuple is train indices and test indices for that fold
    """
    if validation_column is None:
        # For no validation column just sample data randomly
        return [(train_idx, test_idx) for train_idx, test_idx in KFold(n_splits).split(np.arange(num_obs))]
    else:
        groups = [m[validation_column] for m in meta_data]
        return [(train_idx, test_idx) for train_idx, test_idx in GroupKFold(n_splits).split(np.arange(num_obs), groups=groups)]

def split_train_val(num_obs, metadata, tuning_sample_ratio, validation_column=None, val_column_idx=None):
    """
    @param num_obs: number of observations
    @param feat_generator: submotif feature generator
    @param metadata: metadata to include variables to perform validation on
    @param tuning_sample_ratio: ratio of data to place in validation set
    @param validation_column: variable to perform validation on (if None then sample randomly)
    @param val_column_idx: the category we want to force to the validation set (this is mostly for debugging)

    @return training and validation indices
    """
    if validation_column is None:
        # For no validation column just sample data randomly
        val_size = int(tuning_sample_ratio * num_obs)
        if tuning_sample_ratio > 0:
            val_size = max(val_size, 1)
        permuted_idx = np.random.permutation(num_obs)
        train_idx = permuted_idx[:num_obs - val_size]
        val_idx = permuted_idx[num_obs - val_size:]
    else:
        # For a validation column, sample the categories randomly based on
        # tuning_sample_ratio
        categories = set([elt[validation_column] for elt in metadata])
        num_categories = len(categories)
        val_size = int(tuning_sample_ratio * num_categories) + 1
        if tuning_sample_ratio > 0:
            val_size = max(val_size, 1)

        if val_column_idx is None:
            # sample random categories from our validation variable
            val_categories_idx = np.random.choice(len(categories), size=val_size, replace=False)
            val_categories = set([list(categories)[j] for j in val_categories_idx])
        else:
            # choose val_column_idx as validation item
            val_categories = set([list(categories)[val_column_idx]])

        train_categories = categories - val_categories
        print "val cate", val_categories
        print "train cate", train_categories
        train_idx = [idx for idx, elt in enumerate(metadata) if elt[validation_column] in train_categories]
        val_idx = [idx for idx, elt in enumerate(metadata) if elt[validation_column] in val_categories]

    return train_idx, val_idx

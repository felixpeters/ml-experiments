from collections import Counter

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, FunctionTransformer


def corr_matrix(df):
    return df.corr().style.background_gradient(cmap='coolwarm')


def dtypes(df):
    dtypes = [df[feat].dtype.name for feat in df.columns]
    c = Counter(dtypes)
    return c.most_common()


def missing_vals(df, axis=0):
    return df.isnull().sum(axis=axis).sort_values(ascending=False) / df.shape[axis]


def remove_missing_vals(df, cutoff_perc, axis=0):
    missing = df.isnull().sum(axis=axis).sort_values(
        ascending=False) / df.shape[axis]
    drop = missing.loc[missing > cutoff_perc].index.to_list()
    result = df.drop(labels=drop, axis=(1 if axis == 0 else 0))
    return result


def cont_cat_split(df, dep_var=None):
    cont_names, cat_names = [], []
    for label in df:
        if label == dep_var:
            continue
        if df[label].dtype == int or df[label].dtype == float:
            cont_names.append(
                label)
        else:
            cat_names.append(label)
    return cont_names, cat_names


def mark_cat_feats(df, cat_feats):
    cats = list(set(df.columns) & set(cat_feats))
    for cat in cats:
        df[cat] = df[cat].astype('category')
    return df


def impute_missing_vals(df, dep_var=None):
    cont_vars, cat_vars = cont_cat_split(df, dep_var=dep_var)
    cont_imputer = SimpleImputer(missing_values=np.NaN, strategy="median")
    df[cont_vars] = cont_imputer.fit_transform(df[cont_vars])
    cat_imputer = SimpleImputer(
        missing_values=np.NaN, strategy="most_frequent")
    df[cat_vars] = cat_imputer.fit_transform(df[cat_vars])
    return df


def count_cats(df, dep_var=None):
    _, cat_vars = cont_cat_split(df, dep_var=dep_var)
    for cat in cat_vars:
        num_cats = len(pd.unique(df[cat]))
        print(f'Unique values for {cat}: {num_cats}')
    return


def show_cov_top_n_cats(df, n=10, dep_var=None):
    _, cat_vars = cont_cat_split(df, dep_var=dep_var)
    for cat in cat_vars:
        col = df[cat]
        counts = col.value_counts()
        total_count = counts.sum()
        top_n_count = counts[:n].sum()
        print(
            f'Coverage of top {n} categories for feature {col.name}: {top_n_count/total_count*100:.2f}%')
    return


def concat_long_tail_cats(df, n=10, dep_var=None):
    _, cat_vars = cont_cat_split(df, dep_var=dep_var)
    for cat in cat_vars:
        col = df[cat]
        top_n_cats = list(col.value_counts().index[:n])
        mask = [False if row in top_n_cats else True for row in col]
        fill_val = ("other" if str(col.cat.categories.dtype)
                    == "object" else 0)
        col.cat.add_categories(fill_val, inplace=True)
        temp = col.mask(mask, other=fill_val)
        temp.cat.remove_unused_categories(inplace=True)
        df[cat] = temp
    return df


def rescale_cont_vars(df, log_transform=True, dep_var=None):
    cont_vars, _ = cont_cat_split(df, dep_var=dep_var)
    if log_transform:
        log_transformer = FunctionTransformer(
            func=np.log1p, inverse_func=np.expm1, validate=False)
        df[cont_vars] = log_transformer.fit_transform(df[cont_vars])
    scaler = MinMaxScaler()
    df[cont_vars] = scaler.fit_transform(df[cont_vars])
    return df


def one_hot_encode(df, dep_var=None):
    _, cat_vars = cont_cat_split(df, dep_var=dep_var)
    one_hot_df = pd.get_dummies(df[cat_vars], prefix=cat_vars)
    df = df.drop(columns=cat_vars)
    return pd.concat([df, one_hot_df], axis=1)


import numpy as np;

from sklearn.base import BaseEstimator, TransformerMixin
from scipy.sparse import issparse

import spacy


class DenseTransformer(BaseEstimator):
    """Convert a sparse array into a dense array."""

    def __init__(self, return_copy=True):
        self.return_copy = return_copy
        self.is_fitted = False

    def transform(self, X, y=None):
        """ Return a dense version of the input array.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like, shape = [n_samples] (default: None)
        Returns
        ---------
        X_dense : dense version of the input X array.
        """
        if issparse(X):
            return X.toarray()
        elif self.return_copy:
            return X.copy()
        else:
            return X

    def fit(self, X, y=None):
        self.is_fitted = True
        return self

    def fit_transform(self, X, y=None):
        """ Return a dense version of the input array.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like, shape = [n_samples] (default: None)
        Returns
        ---------
        X_dense : dense version of the input X array.
        """
        return self.transform(X=X, y=y)

class POSTagTransformer(BaseEstimator):
    def __init__(self, language='en'):
        self.language = language;

        if language is None:
            raise ValueError('None is not a valid language');
        self.model_ = spacy.load(language.lower(), parser=False, entity=False)

    def transform(self, X, y=None):
        #model_ = taggers[self.language];
        X = [[(
                token.text,
                token.tag_,
                token.pos_,
                token.dep_,
                ) for token in self.model_(doc)] for doc in X];
        return X;       

    def fit(self, X, y=None):
        self.is_fitted = True
        return self

    def fit_transform(self, X, y=None):
        return self.transform(X=X, y=y)
    
    
class FeatureDropper(BaseEstimator, TransformerMixin):
    def __init__(self, drop=[]):
        self.drop = drop

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None, **fit_params):
        return X[list(set(X.columns) - set(self.drop))]
    
    
# this transformer handles missing values 
class FillNaTransformer(BaseEstimator, TransformerMixin):    
    def __init__(self, columns = None):
        self.columns = columns
        
    def fit(self, X, y=None, **fit_params):
        if self.columns is None:
            self.columns = X.select_dtypes(exclude = ["object"]).columns
        self.train_median = X[self.columns].median()
        return self

    def transform(self, X):
        X[self.columns] = X[self.columns].fillna(self.train_median) 
        return X



class FilterTagTransformer(BaseEstimator):
    def __init__(self,token='POS', parts=None):
        self.token = token;
        self.parts = parts;
            

    def transform(self, X, y=None):
        """ Return An array of tokens 
        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_tokens]
            Array documents, where each document consists of a list of node
            and each node consist of a token and its correspondent tag
            
            [
            [('a','TAG1'),('b','TAG2')],
            [('a','TAG1')]
            ]
        y : array-like, shape = [n_samples] (default: None)
        Returns
        ---------
        X_dense : dense version of the input X array.
        """
        if self.token == 'TAG':
            X = [' '.join([d[1].split('__')[0] for d in doc]) for doc in X]
        elif self.token == 'POS':
            if self.parts is None:
                X = [' '.join([d[2] for d in doc]) for doc in X];
            else:
                X = [' '.join([d[0] for d in doc if d[2] in self.parts]) for doc in X]
        elif self.token == 'DEP':
            X = [' '.join([d[3] for d in doc]) for doc in X]
        elif self.token == 'word_POS':
            if self.parts is None:
                X = [' '.join([d[0]+'/'+d[2] for d in doc]) for doc in X]
        elif self.token == 'filter':
            X = [' '.join([d[0] for d in doc if d[2] in self.parts]) for doc in X]
        else:
            X = [' '.join([d[0] for d in doc]) for doc in X]
        
        return np.array(X);       

    def fit(self, X, y=None):
        self.is_fitted = True
        return self

    def fit_transform(self, X, y=None):
        return self.transform(X=X, y=y)
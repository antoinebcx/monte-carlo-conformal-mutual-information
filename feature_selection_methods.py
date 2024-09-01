from sklearn.feature_selection import mutual_info_classif, SelectKBest, chi2, f_classif, RFECV
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SequentialFeatureSelector as SklearnSFS
from sklearn.linear_model import LogisticRegression, LassoCV, ElasticNetCV
from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import SparsePCA
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_random_state
from sklearn.preprocessing import StandardScaler
import numpy as np

class MutualInformationSelect(BaseEstimator, TransformerMixin):
    def __init__(self, k=10):
        self.k = k
        self.selector = None

    def fit(self, X, y):
        self.selector = SelectKBest(mutual_info_classif, k=self.k)
        self.selector.fit(X, y)
        return self

    def transform(self, X):
        return self.selector.transform(X)

    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X)


class ChiSquareSelect(BaseEstimator, TransformerMixin):
    def __init__(self, k=10):
        self.k = k
        self.selector = None

    def fit(self, X, y):
        self.selector = SelectKBest(chi2, k=self.k)
        self.selector.fit(X, y)
        return self

    def transform(self, X):
        return self.selector.transform(X)

    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X)


class FTestSelect(BaseEstimator, TransformerMixin):
    def __init__(self, k=10):
        self.k = k
        self.selector = None

    def fit(self, X, y):
        self.selector = SelectKBest(f_classif, k=self.k)
        self.selector.fit(X, y)
        return self

    def transform(self, X):
        return self.selector.transform(X)

    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X)


class LassoSelect(BaseEstimator, TransformerMixin):
    def __init__(self, alpha=1.0, cv=5):
        self.alpha = alpha
        self.cv = cv
        self.model = None

    def fit(self, X, y):
        self.model = LassoCV(alphas=np.logspace(-4, 1, 50), cv=self.cv).fit(X, y)
        return self

    def transform(self, X):
        selected_features = np.where(self.model.coef_ != 0)[0]
        return X[:, selected_features]

    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X)


class RandomForestSelect(BaseEstimator, TransformerMixin):
    def __init__(self, n_estimators=100, max_features='auto', n_features_to_select=10, random_state=None):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.n_features_to_select = n_features_to_select
        self.random_state = random_state
        self.model = None
        self.selected_features = None

    def fit(self, X, y):
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_features=self.max_features,
            random_state=self.random_state
        )
        self.model.fit(X, y)
        importances = self.model.feature_importances_
        self.selected_features = np.argsort(importances)[-self.n_features_to_select:]
        return self

    def transform(self, X):
        return X[:, self.selected_features]

    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X)


class ElasticNetSelect(BaseEstimator, TransformerMixin):
    def __init__(self, cv=5, l1_ratio=0.5, max_iter=1000, random_state=None):
        self.cv = cv
        self.l1_ratio = l1_ratio
        self.max_iter = max_iter
        self.random_state = random_state
        self.model = None

    def fit(self, X, y):
        self.model = ElasticNetCV(cv=self.cv, l1_ratio=self.l1_ratio, max_iter=self.max_iter, random_state=self.random_state)
        self.model.fit(X, y)
        return self

    def transform(self, X):
        selected_features = np.where(self.model.coef_ != 0)[0]
        return X[:, selected_features]

    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X)
    

class SparsePCASelect(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=100, alpha=1.0, random_state=None):
        self.n_components = n_components
        self.alpha = alpha
        self.random_state = random_state
        self.model = None

    def fit(self, X, y=None):
        self.model = SparsePCA(n_components=self.n_components, alpha=self.alpha, random_state=self.random_state)
        self.model.fit(X)
        return self

    def transform(self, X):
        X_transformed = self.model.transform(X)
        return X_transformed

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)
    

class StabilitySelection(BaseEstimator, TransformerMixin):
    def __init__(self, base_estimator=None, n_iterations=100, sample_fraction=0.75, 
                 threshold=0.5, n_features_to_select=10, random_state=None):
        self.base_estimator = base_estimator or LogisticRegression(random_state=random_state)
        self.n_iterations = n_iterations
        self.sample_fraction = sample_fraction
        self.threshold = threshold
        self.n_features_to_select = n_features_to_select
        self.random_state = random_state

    def fit(self, X, y):
        random_state = check_random_state(self.random_state)
        n_samples, n_features = X.shape
        
        feature_scores = np.zeros(n_features)
        
        for _ in range(self.n_iterations):
            sample_mask = random_state.rand(n_samples) < self.sample_fraction
            X_sample, y_sample = X[sample_mask], y[sample_mask]
            
            selector = SelectKBest(score_func=f_classif, k='all')
            selector.fit(X_sample, y_sample)
            
            feature_scores += (selector.scores_ > np.median(selector.scores_)).astype(int)
        
        self.feature_scores_ = feature_scores / self.n_iterations
        self.selected_features_ = np.where(self.feature_scores_ > self.threshold)[0]
        
        if len(self.selected_features_) < self.n_features_to_select:
            self.selected_features_ = np.argsort(self.feature_scores_)[::-1][:self.n_features_to_select]
        
        return self

    def transform(self, X):
        return X[:, self.selected_features_]


class EnsembleFeatureSelection(BaseEstimator, TransformerMixin):
    def __init__(self, estimators=None, n_features_to_select=10, voting='soft'):
        self.estimators = estimators or [
            ('lasso', LassoCV(random_state=42)),
            ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
            ('lr', LogisticRegression(random_state=42))
        ]
        self.n_features_to_select = n_features_to_select
        self.voting = voting

    def fit(self, X, y):
        self.feature_importances_ = np.zeros(X.shape[1])
        
        for name, estimator in self.estimators:
            selector = SelectFromModel(estimator, max_features=self.n_features_to_select)
            selector.fit(X, y)
            
            if self.voting == 'soft':
                if hasattr(selector.estimator_, 'coef_'):
                    self.feature_importances_ += np.abs(selector.estimator_.coef_[0])
                elif hasattr(selector.estimator_, 'feature_importances_'):
                    self.feature_importances_ += selector.estimator_.feature_importances_
            elif self.voting == 'hard':
                self.feature_importances_ += selector.get_support()
        
        self.selected_features_ = np.argsort(self.feature_importances_)[::-1][:self.n_features_to_select]
        return self

    def transform(self, X):
        return X[:, self.selected_features_]


class BorutaPySelector(BaseEstimator, TransformerMixin):
    def __init__(self, n_estimators='auto', perc=100, alpha=0.05, max_iter=100, random_state=None):
        self.n_estimators = n_estimators
        self.perc = perc
        self.alpha = alpha
        self.max_iter = max_iter
        self.random_state = random_state
        self.boruta = None

    def fit(self, X, y):
        rf = RandomForestClassifier(n_estimators=self.n_estimators, random_state=self.random_state)
        self.boruta = BorutaPy(rf, n_estimators=self.n_estimators, perc=self.perc, 
                               alpha=self.alpha, max_iter=self.max_iter, 
                               random_state=self.random_state)
        self.boruta.fit(X, y)
        return self

    def transform(self, X):
        return self.boruta.transform(X)

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)
    

class ReliefFSelector(BaseEstimator, TransformerMixin):
    def __init__(self, n_neighbors=10, n_features_to_select=10, random_state=None):
        self.n_neighbors = n_neighbors
        self.n_features_to_select = n_features_to_select
        self.random_state = random_state

    def fit(self, X, y):
        random_state = check_random_state(self.random_state)
        n_samples, n_features = X.shape

        X_scaled = StandardScaler().fit_transform(X)
        
        feature_weights = np.zeros(n_features)
        
        for _ in range(n_samples):
            instance = random_state.randint(n_samples)
            near_hit = self._find_near_hit(X_scaled, y, instance)
            near_miss = self._find_near_miss(X_scaled, y, instance)
            
            feature_weights += np.abs(X_scaled[instance] - X_scaled[near_hit]) - \
                               np.abs(X_scaled[instance] - X_scaled[near_miss])
        
        self.feature_weights_ = feature_weights / n_samples
        self.selected_features_ = np.argsort(self.feature_weights_)[::-1][:self.n_features_to_select]
        return self

    def transform(self, X):
        return X[:, self.selected_features_]

    def _find_near_hit(self, X, y, instance):
        same_class = np.where(y == y[instance])[0]
        distances = np.sum((X[same_class] - X[instance])**2, axis=1)
        return same_class[np.argsort(distances)[1]]  # Exclude the instance itself

    def _find_near_miss(self, X, y, instance):
        different_class = np.where(y != y[instance])[0]
        distances = np.sum((X[different_class] - X[instance])**2, axis=1)
        return different_class[np.argmin(distances)]


class RFECVSelector(BaseEstimator, TransformerMixin):
    def __init__(self, estimator=None, step=1, cv=5, scoring='accuracy', n_jobs=-1):
        self.estimator = estimator or LogisticRegression(random_state=42)
        self.step = step
        self.cv = cv
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.selector = None

    def fit(self, X, y):
        self.selector = RFECV(estimator=self.estimator, step=self.step, cv=self.cv, 
                              scoring=self.scoring, n_jobs=self.n_jobs)
        self.selector.fit(X, y)
        return self

    def transform(self, X):
        return self.selector.transform(X)


class SequentialFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, estimator=None, n_features_to_select=10, direction='forward', scoring='accuracy', cv=5, n_jobs=-1):
        self.estimator = estimator or LogisticRegression(random_state=42)
        self.n_features_to_select = n_features_to_select
        self.direction = direction
        self.scoring = scoring
        self.cv = cv
        self.n_jobs = n_jobs
        self.selector = None

    def fit(self, X, y):
        self.selector = SklearnSFS(
            estimator=self.estimator,
            n_features_to_select=self.n_features_to_select,
            direction=self.direction,
            scoring=self.scoring,
            cv=self.cv,
            n_jobs=self.n_jobs
        )
        self.selector.fit(X, y)
        return self

    def transform(self, X):
        return self.selector.transform(X)

    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X)
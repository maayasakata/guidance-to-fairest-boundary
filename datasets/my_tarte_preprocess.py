import pandas as pd
from sklearn.preprocessing import PowerTransformer, RobustScaler
from sklearn.pipeline import Pipeline
from skrub import to_datetime
from tarte_ai.tarte_preprocess_table import TARTE_TablePreprocessor as _BasePreproc


class MyTARTE_TablePreprocessor(_BasePreproc):
    def fit(self, X, y=None):
        """
        Fit function used for the TARTE_TablePreprocessor.

        Parameters
        ----------
        X : pandas.DataFrame
            Input data to fit.
        y : array-like, optional
            Target values, by default None.

        Returns
        -------
        self : TARTE_TablePreprocessor
            Fitted transformer.
        """

        if isinstance(X, pd.DataFrame) == False:
            X = pd.DataFrame(X)
            col_names = [f"Column_{i}" for i in range(X.shape[1])]
            X = X.set_axis(col_names, axis="columns")

        X_ = X.replace("\n", " ", regex=True).copy()
        self.is_fitted_ = False
        self.y_ = y

        # Load language_model
        if not hasattr(self, "lm_model_"):
            self._load_lm_model()

        # Preprocess for Datetime information
        dat_col_names = []
        for col in X_:
            if pd.api.types.is_datetime64_any_dtype(to_datetime(X_[col])):
                datetime = pd.to_datetime(X_[col].astype("datetime64[s]"))
                X_[col] = datetime.dt.strftime("%Y").astype(float)
                dat_col_names.append(col)
        self.dat_col_names_ = dat_col_names

        # Use original column names without lowercasing to avoid mismatches
        cat_col_names = X_.select_dtypes(include="object").columns.str.replace(
            "\n", " ", regex=True
        )
        self.cat_col_names_ = list(set(cat_col_names) - set(dat_col_names))

        num_col_names = X_.select_dtypes(exclude="object").columns.str.replace(
            "\n", " ", regex=True
        )
        self.num_col_names_ = list(set(num_col_names) - set(dat_col_names))
        self.col_names_ = (
            self.cat_col_names_ + self.num_col_names_ + self.dat_col_names_
        )

        # Set max-pad-size
        self.max_pad_size_ = len(self.col_names_)

        # Set transformers for numerical and datetime
        self.num_transformer_ = RobustScaler().set_output(transform="pandas")
        self.dat_transformer_ = Pipeline([("scale", RobustScaler()), ("power", PowerTransformer()),]).set_output(transform="pandas")

        # Ensure numerical columns exist before fitting the transformer
        num_cols_exist = [col for col in self.num_col_names_ if col in X_.columns]
        if num_cols_exist:
            self.num_transformer_.fit(X_[num_cols_exist])

        dat_cols_exist = [col for col in self.dat_col_names_ if col in X_.columns]
        if dat_cols_exist:
            self.dat_transformer_.fit(X_[dat_cols_exist])

        self.is_fitted_ = True
        return self
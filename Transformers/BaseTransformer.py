class BaseTransformer():
    def __init__():
        self.required_keys = None

    def fit(self, df):
        for key in self.required_keys:
            df[key]
        return self

    def fit_transform(self, df):
        self.fit(df)
        transformed_df = self.transform(df)
        return transformed_df

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import plotly.express as px


class PCA_Analysis:
    def _perform_pca(self, df: pd.DataFrame) -> list:
        """
            Private method to perform PCA analysis on the dataframe
            and return the dataframe with the PCA components
        """
        scaler = StandardScaler()
        scaled_df = scaler.fit_transform(df)
        pca = PCA(n_components=2)
        pca_components = pca.fit_transform(scaled_df)
        pca_df = pd.DataFrame(data=pca_components, columns=["PC1", "PC2"])
        return [pca_df, pca]

    def pca_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """
            This function performs PCA analysis on the dataframe
            and returns the dataframe with the PCA components
        """
        return self._perform_pca(df)[0]

    def loadings(self, df: pd.DataFrame) -> pd.DataFrame:
        """
            This function returns the loadings for the PCA components
        """
        pca = self._perform_pca(df)[1]

        loadings = pd.DataFrame(
            pca.components_.T, columns=["PC1", "PC2"], index=df.columns
        )

        return loadings

    def top_attributes(self, loadings: pd.DataFrame, n: int) -> dict:
        """
            This function returns the top n components
        """
        top_attributes_pc1 = loadings["PC1"].abs().sort_values(ascending=False).head(n)
        top_attributes_pc2 = loadings["PC2"].abs().sort_values(ascending=False).head(n)

        # top_attributes_pc1 = top_attributes_pc1.head(n)
        # top_attributes_pc2 = top_attributes_pc2.head(n)

        struct_attributes = {}
        struct_attributes["PC1"] = top_attributes_pc1.index.tolist()
        struct_attributes["PC2"] = top_attributes_pc2.index.tolist()

        return struct_attributes

    def explained_variance(self, df) -> pd.DataFrame:
        """
            This function returns the explained variance for the PCA components
        """

        pca = self._perform_pca(df)[1]

        ev = pca.explained_variance_ratio_
        explained_variance = np.insert(ev, 0, 0)
        cum_variance = np.cumsum(np.round(explained_variance, decimals=3))

        pca_df = pd.DataFrame(["", "PC1", "PC2"], columns=["PC"])
        explained_variance_df = pd.DataFrame(
            explained_variance, columns=["Explained Variance"]
        )
        cum_variance_df = pd.DataFrame(cum_variance, columns=["Cumulative Variance"])
        df_explained_variance = pd.concat(
            [pca_df, explained_variance_df, cum_variance_df], axis=1
        )

        return df_explained_variance

    def pca_plot(self, df: pd.DataFrame) -> None:
        """
            This function plots the PCA components
        """

        fig = px.bar(
            df, x="PC", y="Explained Variance", text="Explained Variance", width=800
        )
        fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")

        return fig.show()

    # def pca_plot_3d(self, df: pd.DataFrame) -> None:
    #     """
    #     This function plots the PCA components
    #     """
    #     loadings_label = df.index

    #     fig = px.scatter_3d(
    #         df, x="PC1", y="PC2", z="PC3", text=loadings_label, width=800
    #     )
    #     return fig.show()

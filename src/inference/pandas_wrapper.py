"""
This module provides tools to lazily load and manage data in a pandas DataFrame-like structure using proxies.

Classes:
    Proxy: A class for lazy loading objects.
    PandasWrapper: A class for wrapping a dictionary of data into a pandas DataFrame-like structure with lazy loading.
    _LocIndexer: Helper class for PandasWrapper to support .loc indexing with lazy loading.
    _iLocIndexer: Helper class for PandasWrapper to support .iloc indexing with lazy loading.
"""

import pandas as pd
from typing import Tuple

class Proxy:
    """
    A proxy class for lazy loading an object.

    Attributes:
        loader (callable): A callable that loads the object.
        _object: The lazily loaded object.
    """
    
    def __init__(self, loader):
        self.loader = loader
        self._object = None

    @property
    def object(self):
        """
        Returns the loaded object, loading it if necessary.
        """
        if self._object is None:
            self._object = self.loader()
        return self._object

    def __repr__(self):
        """
        Returns a string representation of the Proxy.
        """
        return f"LazyLoaded: {self._object is not None}"


class PandasWrapper:
    """
    A class for wrapping a dictionary of data into a pandas DataFrame-like structure with lazy loading.

    Attributes:
        data (dict): The input data.
        object_name (str): The name of the object in the DataFrame.
        df (pd.DataFrame): The pandas DataFrame storing the proxies.
    """
    
    def __init__(self, data: dict, object_name: str):
        self.data = data
        self.object_name = object_name
        self.df = self._to_dataframe(data)

    def _to_dataframe(self, data: dict) -> pd.DataFrame:
        """
        Converts the input data dictionary into a pandas DataFrame with proxies.

        Args:
            data (dict): The input data.

        Returns:
            pd.DataFrame: The DataFrame with proxies.
        """
        index = pd.MultiIndex.from_tuples(data.keys(), names=["ID", "IDPix"])
        proxies = [Proxy(lambda key=key: self._load_object(key)) for key in data.keys()]
        df = pd.DataFrame(proxies, index=index, columns=[self.object_name])
        return df

    def _load_object(self, key: Tuple[int, int]):
        """
        Loads the object corresponding to the given key.

        Args:
            key (Tuple[int, int]): The key for the object.

        Returns:
            The loaded object.
        """
        return self.data[key]

    def __getitem__(self, key):
        """
        Gets the object corresponding to the given key, loading it if necessary.

        Args:
            key: The key for the object.

        Returns:
            The loaded object.
        """
        result = self.df.loc[key, self.object_name]
        if isinstance(result, pd.Series):
            return result.apply(lambda x: x.object)
        return result.object

    def __setitem__(self, key, value):
        """
        Sets the object corresponding to the given key.

        Args:
            key: The key for the object.
            value: The value to set.
        """
        self.df.loc[key, self.object_name]._object = value

    def __repr__(self):
        """
        Returns a string representation of the PandasWrapper.
        """
        return repr(self.df)

    @property
    def index(self):
        """
        Returns the index of the DataFrame.
        """
        return self.df.index

    @property
    def iloc(self):
        """
        Returns the _iLocIndexer for integer-location based indexing.
        """
        return _iLocIndexer(self)

    @property
    def loc(self):
        """
        Returns the _LocIndexer for label based indexing.
        """
        return _LocIndexer(self)


class _LocIndexer:
    """
    Helper class for PandasWrapper to support .loc indexing with lazy loading.
    """
    
    def __init__(self, parent):
        self.parent = parent

    def __getitem__(self, key):
        """
        Gets the object corresponding to the given key using .loc indexing.

        Args:
            key: The key for the object.

        Returns:
            The loaded object.
        """
        result = self.parent.df.loc[key, self.parent.object_name]
        if isinstance(result, pd.Series):
            return result.apply(lambda x: x.object)
        return result.object

    def __setitem__(self, key, value):
        """
        Sets the object corresponding to the given key using .loc indexing.

        Args:
            key: The key for the object.
            value: The value to set.
        """
        self.parent.df.loc[key, self.parent.object_name]._object = value


class _iLocIndexer:
    """
    Helper class for PandasWrapper to support .iloc indexing with lazy loading.
    """
    
    def __init__(self, parent):
        self.parent = parent

    def __getitem__(self, key):
        """
        Gets the object corresponding to the given key using .iloc indexing.

        Args:
            key: The key for the object.

        Returns:
            The loaded object.
        """
        result = self.parent.df.iloc[key, self.parent.df.columns.get_loc(self.parent.object_name)]
        if isinstance(result, pd.Series):
            return result.apply(lambda x: x.object)
        return result.object

    def __setitem__(self, key, value):
        """
        Sets the object corresponding to the given key using .iloc indexing.

        Args:
            key: The key for the object.
            value: The value to set.
        """
        self.parent.df.iloc[key, self.parent.df.columns.get_loc(self.parent.object_name)]._object = value

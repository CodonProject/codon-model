from typing import Any

class CodonDataset:
    '''
    Base class for all Codon datasets.

    This abstract class defines the interface that all datasets must implement.
    It provides a common structure for accessing data rows and length.
    '''

    @property
    def row(self) -> int:
        '''
        Returns the number of rows in the dataset.

        Returns:
            int: The total number of rows.
        '''
        return len(self)

    def __len__(self) -> int:
        '''
        Returns the length of the dataset.

        Returns:
            int: The length of the dataset.

        Raises:
            NotImplementedError: If the method is not implemented by the subclass.
        '''
        raise NotImplementedError

    def __getitem__(self, idx: Any) -> Any:
        '''
        Retrieves an item from the dataset at the specified index.

        Args:
            idx (Any): The index of the item to retrieve.

        Returns:
            Any: The item at the specified index.

        Raises:
            NotImplementedError: If the method is not implemented by the subclass.
        '''
        raise NotImplementedError
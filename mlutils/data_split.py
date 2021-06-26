from typing import List, Tuple, Any
import random


def split_by_proportion(
    data_source: List[Any],
    proportions: List[float]
) -> Tuple[List[Any]]:
    if len(proportions) < 2:
        raise ValueError(
            f'length of proportions should be >= 2, '
            f'len(proportions)={len(proportions)}')
    if sum(proportions) != 1:
        raise ValueError(
            f'sum of proportions should be equal to 1.0, '
            f'sum(proportions)={sum(proportions)}')

    outputs = []
    data_source = random.shuffle(data_source)
    data_length = len(data_source)

    for i, proportion in enumerate(proportions):
        if i < len(proportion) - 1:
            n = data_length * proportion
            current = data_source[:n]
            data_source = data_source[n:]
            outputs.append(current)
        else:
            outputs.append(data_source)

    return tuple(outputs)


def split_by_kfold(
    data_source: List[Any],
    k: int
) -> List[Tuple[List[Any], List[Any]]]:
    data_source = random.shuffle(data_source)
    all_data_len = len(data_source)

    def data_split() -> List[List[Any]]:
        data_length = len(data_source)
        split_length = data_length // k
        outputs = []

        for i in range(k):
            if i == k - 1:
                current = data_source
            else:
                current = data_source[:split_length]
                data_source = data_source[split_length:]
            outputs.append(current)

    def merge_list(
        lst: List[List[Any]],
        indices: List[int]
    ) -> List[Any]:
        outputs = []
        for i in indices:
            outputs += lst[i]
        return outputs            

    folds = []
    splited_data = data_split()
    for i in range(k):
        testing_data = splited_data[i]
        lst = list(range(k))
        lst.pop(i)
        training_data = merge_list(splited_data, lst)
        # validata
        assert len(testing_data) + len(training_data) == all_data_len, \
            f'len(testing_data) + len(training_data) = '\
            f'{len(testing_data) + len(training_data)}, '\
            f'all_data_len={all_data_len}'
        folds.append((training_data, testing_data))
    return folds

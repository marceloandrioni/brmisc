#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__all__ = ["ListOfObjects"]


from typing import Optional
import time
from functools import singledispatchmethod

import logging


def get_log():
    # use GMT/Zulu time
    logging.Formatter.converter = time.gmtime
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s",
                        datefmt="%Y-%m-%dT%H:%M:%SZ",
                        force=True)
    log = logging.getLogger(__name__)
    log.setLevel(logging.INFO)

    return log


log = get_log()


class ListOfObjects(list):

    # method: signature
    # append: lst.append(object, /)
    # clear: lst.clear()
    # py: lst.copy()
    # count: lst.count(value, /)
    # extend: lst.extend(iterable, /)
    # index: lst.index(value, start=0, stop=9223372036854775807, /)
    # insert: lst.insert(index, object, /)
    # pop: lst.pop(index=-1, /)
    # remove: lst.remove(value, /)
    # reverse: lst.reverse()
    # sort: lst.sort(*, key=None, reverse=False)

    def __init__(self, iterable=(), id_field: Optional[str] = None) -> None:
        """
        List of objects.

        Parameters
        ----------
        iterable : iterable
            List of objects. Object can have distinct types, but all objects
            must have an attribute/key that will be used as an unique
            identifier (`id_field`).
        id_field : str
            Attribute/key used as an unique identifier.

        Returns
        -------
        out : ListOfObjects instance

        Examples
        --------
        >>> @dataclass
        ... class Vehicle:
        ...     type: str
        ...     maker: str
        ...     model: str
        ...     year: int

        >>> lst = [Vehicle("car", "Honda", "Civic", 2020),
        ...        Vehicle("truck", "Scania", "770 V8", 2019),
        ...        dict(type="motorcycle", maker="Honda", model="CG 150", year=2021)]

        >>> ListOfObjects(lst, id_field="type")
        car: Vehicle(type='car', maker='Honda', model='Civic', year=2020),
        truck: Vehicle(type='truck', maker='Scania', model='770 V8', year=2019),
        motorcycle: {'type': 'motorcycle', 'maker': 'Honda', 'model': 'CG 150', 'year': 2021}

        >>> ListOfObjects(lst, id_field="maker")
        ValueError: All elements must have a unique value for the attribute/key
        'maker'. The value 'Honda' appears 2 times.

        """

        super().__init__(iterable)

        if not isinstance(id_field, str):
            raise ValueError("id_field must be a string")

        self._id_field = id_field

        self.ids

    def __repr__(self) -> str:
        """Return repr(self)."""

        if len(self) == 0:
            return "<empty>"

        return ",\n".join(f"{id}: {obj}" for id, obj in zip(self.ids, self))

    @property
    def id_field(self) -> str:
        return self._id_field

    @singledispatchmethod
    def _get_id_field(self, obj) -> str:
        try:
            return getattr(obj, self.id_field)
        except AttributeError as e:
            raise e.__class__(f"{e} in {obj}")

    @_get_id_field.register(dict)
    def _(self, obj) -> str:
        try:
            return obj[self.id_field]
        except KeyError as e:
            raise e.__class__(f"{e} in {obj}")

    @property
    def ids(self) -> list:

        ids = [self._get_id_field(obj) for obj in self]

        if len(set(ids)) != len(ids):

            for id in ids:
                n = ids.count(id)
                if n > 1:
                    break

            raise ValueError(
                "All elements must have a unique value for the attribute/key"
                f" '{self.id_field}'. The value '{id}' appears {n} times.")

        return ids

    def append(self, obj) -> None:
        super().append(obj)
        self.ids

    # clear: same
    # copy: same

    def count(self, id) -> int:
        return self.ids.count(id)

    def extend(self, iterable) -> None:
        super().extend(iterable)
        self.ids

    def index(self, id, *args, **kwargs) -> int:
        return self.ids.index(id, *args, **kwargs)

    # insert: same

    def insert_before_id(self, id, obj) -> None:
        super().insert(self.index(id), obj)
        self.ids

    def insert_after_id(self, id, obj) -> None:
        super().insert(self.index(id) + 1, obj)
        self.ids

    # pop: same

    def pop_id(self, id):
        return super().pop(self.index(id))

    def remove(self, id) -> None:
        super().remove(self[self.index(id)])

    # reverse: same

    def sort(self, key=None, reverse=False) -> None:
        """
        Sort the list in ascending order and return None.

        The sort is in-place (i.e. the list itself is modified) and stable (i.e.
        the order of two equal elements is maintained). The sort may raise
        an Exception if the the chosen sorting field (e.g.: `id_field`) has
        distinct types.

        Parameters
        ----------
        key : callable, optional
            If a key function is given, apply it once to each list item and
            sort them, ascending or descending, according to their function
            values. If None, sort by `id_field` values.
        reverse : bool, optional
            The reverse flag can be set to sort in descending order.

        Returns
        -------
        None

        Examples
        --------
        >>> vehicles = ListOfObjects(lst, id_field="type")

        default sort (by `id_field`)

        >>> vehicles.sort()

        sort by other attribute

        >>> vehicles.sort(key=lambda x: x.year)

        """

        key = self._get_id_field if key is None else key
        super().sort(key=key, reverse=reverse)

    def get(self, id):
        return self[self.ids.index(id)]

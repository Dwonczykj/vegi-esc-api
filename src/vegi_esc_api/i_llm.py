from __future__ import annotations
from typing import Protocol, Self, Type
from flask import Flask
import argparse


class I_Am_LLM(Protocol):
    @property
    def model(self) -> Self:
        ...

    @classmethod
    def getModel(cls: Type[Self], app: Flask, args: argparse.Namespace | None = None) -> Self:
        '''
        Takes care of downloading, initialisation, singleton fetching and is a factory method that returns an object of this class
        '''
        ...
        
    def most_similar_esc_products(self, *vegi_product_names: str) -> dict | None:
        ...

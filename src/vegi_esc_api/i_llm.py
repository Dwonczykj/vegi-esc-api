from __future__ import annotations
from typing import Protocol, Self, Type, Any
from flask import Flask
import argparse


class I_Am_LLM(Protocol):
    @property
    def model(self) -> Self:
        ...
    
    @property
    def vocab(self) -> list[str]:
        ...
        
    @property
    def key_to_index(self) -> dict[str, str]:
        ...
        
    # @property
    # def key_to_index(self) -> list[str]:
    #     ...
    
    def __getitem__(self, document_id: str) -> Any | None:
        ...
                
    def most_similar(self, words: list[str], top_n: int = 1) -> list[str]:
        ...
        
    def similarity(self, words: list[str], measure_similarity_to_word: str) -> dict[str, float]:
        '''
        inverse of distance (defaults to cosine)
        @returns `dict[str, float]` where the keys are the words that we are comparing similarity of `measure_similarity_to_word` with and the values are the similarity measure
        '''
        ...
        
    def distance(self, words: list[str], measure_similarity_to_word: str) -> dict[str, float]:
        '''inverse of similarity (defaults to cosine)
        @returns `dict[str, float]` where the keys are the words that we are comparing similarity of `measure_similarity_to_word` with and the values are the distance measure
        '''
        ...

    @classmethod
    def getModel(cls: Type[Self], app: Flask, args: argparse.Namespace | None = None) -> Self:
        '''
        Takes care of downloading, initialisation, singleton fetching and is a factory method that returns an object of this class
        '''
        ...
        
    def most_similar_esc_products(self, *vegi_product_names: str) -> dict | None:
        ...

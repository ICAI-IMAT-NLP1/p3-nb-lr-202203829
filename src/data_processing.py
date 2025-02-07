from typing import List, Dict
from collections import Counter
import torch

try:
    from src.utils import SentimentExample, tokenize
except ImportError:
    from utils import SentimentExample, tokenize


def read_sentiment_examples(infile: str) -> List[SentimentExample]:
    """
    Reads sentiment examples from a file.

    Args:
        infile: Path to the file to read from.

    Returns:
        A list of SentimentExample objects parsed from the file.
    """
    examples: List[SentimentExample] = []
    
    with open(infile, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()  # Eliminar espacios en blanco y saltos de línea
            if not line:  
                continue  # Omitir líneas vacías

            extraction = line.split("\t")

            if len(extraction) != 2:
                print(f"Warning: línea mal formateada -> {line}")
                continue  # Omitir líneas incorrectas

            words = extraction[0].split(" ")

            try:
                label = int(extraction[1])  # Convertir a entero
            except ValueError:
                print(f"Warning: etiqueta inválida en la línea -> {line}")
                continue

            examples.append(SentimentExample(words=words, label=label))

    return examples


def build_vocab(examples: List[SentimentExample]) -> Dict[str, int]:
    """
    Creates a vocabulary from a list of SentimentExample objects.

    The vocabulary is a dictionary where keys are unique words from the examples and values are their corresponding indices.

    Args:
        examples (List[SentimentExample]): A list of SentimentExample objects.

    Returns:
        Dict[str, int]: A dictionary representing the vocabulary, where each word is mapped to a unique index.
    """
    # TODO: Count unique words in all the examples from the training set
    vocab: Dict[str, int] = {}
    counter=0
    for example in examples:
        for word in example.words:
            if word in vocab.keys():
                pass
            else:
                vocab[word] = counter
                counter+=1
    return vocab


def bag_of_words(
    text: List[str], vocab: Dict[str, int], binary: bool = False
) -> torch.Tensor:
    """
    Converts a list of words into a bag-of-words vector based on the provided vocabulary.
    Supports both binary and full (frequency-based) bag-of-words representations.

    Args:
        text (List[str]): A list of words to be vectorized.
        vocab (Dict[str, int]): A dictionary representing the vocabulary with words as keys and indices as values.
        binary (bool): If True, use binary BoW representation; otherwise, use full BoW representation.

    Returns:
        torch.Tensor: A tensor representing the bag-of-words vector.
    """
    # TODO: Converts list of words into BoW, take into account the binary vs full
    bow: torch.Tensor = torch.zeros(len(vocab.keys()))
    if binary:
        for word in text:
            if word in vocab.keys():
                bow[vocab[word]] = 1
    else:
        for word in text:
            if word in vocab.keys():
                bow[vocab[word]]+=1

    return bow

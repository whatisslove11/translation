import torch
from typing import List, Tuple
from unidecode import unidecode
from additional import clear_text, tokenize


# нужно ли засовывать токенизаторы в класс, если токенизация идет не только в датасете, а сами функции не методы класса?

class TranslationDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            de_tokenizer,
            en_tokenizer,
            data
    ) -> None:
        self.data = data # only wmt datasets is allowed
        self.en_tokenizer = en_tokenizer
        self.de_tokenizer = de_tokenizer

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[List[int], List[int]]:
        en_sentence = self.data[idx]['en']
        de_sentence = unidecode(self.data[idx]['de'])

        clear_en = clear_text(en_sentence)
        clear_de = clear_text(de_sentence)

        en_tokens = tokenize(clear_en, self.en_tokenizer)
        de_tokens = tokenize(clear_de, self.de_tokenizer)

        return de_tokens, en_tokens

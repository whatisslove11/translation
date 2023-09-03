import torch
from unidecode import unidecode
import youtokentome as yttm
from additional import clear_text, tokenize
from modules.config import device, CFG


def decode(
        src: str,
        model,
        de_tokenizer: yttm.BPE,
        en_tokenizer: yttm.BPE,
        max_len: int = 512
) -> str:
    """
    Gets string in deutsch and translate it to english
    :param src: string in deutsch
    :param model: TranslateModel, transformer model-translator
    :param de_tokenizer: BPE tokenizer for deutsch
    :param en_tokenizer: BPE tokenizer for english
    :param max_len: max number of tokens in result
    :return: str, translated string in english
    """
    model.eval()
    de_sentence = unidecode(src)
    clear_de = clear_text(de_sentence)
    de_tokens = tokenize(clear_de, de_tokenizer)
    src_tensor = torch.tensor(de_tokens).unsqueeze(0).to(device)
    src_mask = model.make_src_mask(src_tensor).to(device)

    encoded_src = model.encoder(src_tensor, src_mask)

    trg_ids = [1]  # 1 in trg_idx - BOS_token
    while len(trg_ids) <= max_len:
        trg_tensor = torch.tensor(trg_ids).unsqueeze(0).to(device)
        trg_mask = model.make_trg_mask(trg_tensor)

        with torch.no_grad():
            predictions = model.decoder(trg_tensor, encoded_src, src_mask, trg_mask)
            last_pred_id = predictions[:, -1, :].argmax(-1).item()

            if last_pred_id == CFG.EOS_token:
                break

            trg_ids.append(last_pred_id)
    # return trg_ids
    return en_tokenizer.decode(trg_ids, ignore_ids=[CFG.EOS_token, CFG.BOS_token])[0]

# Very good. Your training process is now complete 👍
# Next, I'll give you a script specifically designed for inference, using the trained checkpoint to conduct text translation tests.
#
# I will write it according to your project structure (compatible with your current NMTModel/vocab/checkpoint).
# ✅ I. Reasoning Script：infer-nep-eng.py

import torch
import argparse
import json

from model import NMTModel
from checkpoint import load_checkpoint
import utils


def load_vocab(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


class Lang:
    def __init__(self, vocab_json):

        if "word2index" in vocab_json:
            self.word2index = vocab_json["word2index"]
            self.index2word = {v: k for k, v in self.word2index.items()}

            # ✅ input vocab → There is no SOS/EOS
            self.n_words = len(self.word2index) + 2

        elif "index2word" in vocab_json:
            self.index2word = {int(k): v for k, v in vocab_json["index2word"].items()}
            self.word2index = {v: k for k, v in self.index2word.items()}

            # ✅ output vocab → It already includes SOS/EOS
            self.n_words = len(self.word2index)

        else:
            raise ValueError("❌ vocab incorrect format")


def sentence_to_tensor(lang, sentence, device):
    indexes = [lang.word2index.get(word, 0) for word in sentence.split(' ')]
    indexes.append(1)  # EOS
    return torch.tensor(indexes, dtype=torch.long, device=device).unsqueeze(0)


def decode_output(output, lang):
    _, topi = output.topk(1)
    decoded_ids = topi.squeeze().tolist()

    words = []
    for idx in decoded_ids:
        if idx == 1:  # EOS
            break
        words.append(lang.index2word.get(idx, "<UNK>"))

    return " ".join(words)


def translate(model, sentence, input_lang, output_lang, device):
    model.eval()

    input_tensor = sentence_to_tensor(input_lang, sentence, device)

    with torch.no_grad():
        decoder_out, _, _ = model(input_tensor, None)

    output_sentence = decode_output(decoder_out, output_lang)

    return output_sentence


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # load vocab
    input_vocab = load_vocab(args.input_vocab)
    output_vocab = load_vocab(args.output_vocab)

    input_lang = Lang(input_vocab)
    output_lang = Lang(output_vocab)

    # Initialize the model (it must be consistent with that during training)
    model = NMTModel(
        input_size=input_lang.n_words,
        output_size=output_lang.n_words,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        max_len=args.max_len,
        bidirection=args.bidirection,
        dropout_rate=0.2,
        attention=args.attention,
        model_type=args.model_type,
        device=device
    ).to(device)

    # load checkpoint
    encoder_state_dict, decoder_state_dict = load_checkpoint(args.checkpoint)
    model.encoder.load_state_dict(encoder_state_dict)
    model.decoder.load_state_dict(decoder_state_dict)

    print("✅ Model loaded successfully!")

    # Interactive translation
    while True:
        sentence = input("\nEnter the sentence (enter q to exit): ")
        if sentence.lower() == 'q':
            break

        result = translate(model, sentence, input_lang, output_lang, device)
        print("translation result:", result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--checkpoint', required=True, type=str)
    parser.add_argument('--input_vocab', default='input_vocab.json')
    parser.add_argument('--output_vocab', default='output_vocab.json')

    # Model parameters (must be consistent with the training)
    parser.add_argument('--hidden_size', default=128, type=int)
    parser.add_argument('--num_layers', default=2, type=int)
    parser.add_argument('--max_len', default=16, type=int)
    parser.add_argument('--bidirection', action='store_true')
    parser.add_argument('--attention', action='store_true')
    parser.add_argument('--model_type', default='lstm', type=str)

    parser.add_argument('--device', default='cuda', type=str)

    args = parser.parse_args()
    main(args)

    # ✅ II. Operating Mode
    # python infer-nep-eng.py \
    # --checkpoint saved_checkpoint_nep_to_eng/nmt-epoch=21-val_loss=1.793-val_quad_bleu=0.000.ckpt \
    # --input_vocab vocab_nep_to_eng/input_vocab.json \
    # --output_vocab vocab_nep_to_eng/output_vocab.json \
    # --hidden_size 128 \
    # --num_layers 2 \
    # --max_len 16 \
    # --attention \
    # --bidirection
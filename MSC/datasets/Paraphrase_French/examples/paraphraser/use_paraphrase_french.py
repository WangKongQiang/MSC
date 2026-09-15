#!/usr/bin/env python3 -u

import argparse
import csv
import fileinput
import logging
import os
import sys

from fairseq.models.transformer import TransformerModel

logging.getLogger().setLevel(logging.INFO)


def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--en2fr", required=True, help="path to en2fr model")
    parser.add_argument(
        "--fr2en", required=True, help="path to fr2en mixture of experts model"
    )
    parser.add_argument(
        "--user-dir", help="path to fairseq examples/translation_moe/src directory"
    )
    parser.add_argument(
        "--num-experts",
        type=int,
        default=10,
        help="(keep at 10 unless using a different model)",
    )
    parser.add_argument(
        "files",
        nargs="*",
        default=["-"],
        help='input files to paraphrase; "-" for stdin',
    )
    args = parser.parse_args()

    if args.user_dir is None:
        args.user_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),  # examples/
            "translation_moe",
            "src",
        )
        if os.path.exists(args.user_dir):
            logging.info("found user_dir:" + args.user_dir)
        else:
            raise RuntimeError(
                "cannot find fairseq examples/translation_moe/src "
                "(tried looking here: {})".format(args.user_dir)
            )

    logging.info("loading en2fr model from:" + args.en2fr)
    en2fr = TransformerModel.from_pretrained(
        model_name_or_path=args.en2fr,
        tokenizer="moses",
        bpe="sentencepiece",
    ).eval()

    logging.info("loading fr2en model from:" + args.fr2en)
    fr2en = TransformerModel.from_pretrained(
        model_name_or_path=args.fr2en,
        checkpoint_file="model.pt.fixed_with_cfg.pt",
        tokenizer="moses",
        bpe="sentencepiece",
        user_dir=args.user_dir,
        task="translation_moe",
    ).eval()

    def gen_paraphrases(en):
        fr = en2fr.translate(en)
        return [
            fr2en.translate(fr, inference_step_args={"expert": i})
            for i in range(args.num_experts)
        ]

    # Modified part: Read the CSV file, generate paraphrases, and write them to a new CSV file.
    input_file = args.files[0]  # Suppose the input file name is the value in the parameter args.files
    print("args.files:", args.files[0])

    output_file = "./examples/paraphraser/agnews_paraphrase.csv"  # export file name
    # output_file = "./examples/paraphraser/searchsnippets_paraphrase.csv"  # export file name
    # output_file = "./examples/paraphraser/stackoverflow_paraphrase.csv"  # export file name
    # output_file = "./examples/paraphraser/biomedical_paraphrase.csv"  # export file name
    # output_file = "./examples/paraphraser/googlenews_TS_paraphrase.csv"  # export file name
    # output_file = "./examples/paraphraser/googlenews_T_paraphrase.csv"  # export file name
    # output_file = "./examples/paraphraser/googlenews_S_paraphrase.csv"  # export file name
    # output_file = "./examples/paraphraser/tweet_paraphrase.csv"  # export file name

    with open(input_file, mode='r', encoding='ISO-8859-1') as infile, \
            open(output_file, mode='w', encoding='utf-8', newline='') as outfile:

        reader = csv.DictReader(infile)
        writer = csv.writer(outfile)

        # Write to header
        writer.writerow(['label', 'text', 'text1', 'text2'])

        for row in reader:
            label = row['label']
            text = row['text'].strip()
            if len(text) == 0:
                continue
            paraphrases = gen_paraphrases(text)
            # Assuming num-experts = 2, only the first two will be taken.
            text1 = paraphrases[0] if len(paraphrases) > 0 else ""
            text2 = paraphrases[1] if len(paraphrases) > 1 else ""
            writer.writerow([label, text, text1, text2])

    logging.info(f"Paraphrased CSV saved to {output_file}")


if __name__ == "__main__":
    main()

# Switch to the local file directory
# cd /mnt/c/Users/8888/PycharmProjects/pythonProject14/sccl/AugData/fairseq

# Activate the virtual environment
# conda activate sccl-fairseq

# Using Scripts
# python examples/paraphraser/use_paraphrase_french.py   --en2fr paraphraser.en-fr   --fr2en paraphraser.fr-en.hMoEup   --user-dir examples/translation_moe/translation_moe_src --num-experts 2  ./examples/paraphraser/agnews.csv
# python examples/paraphraser/use_paraphrase_french.py   --en2fr paraphraser.en-fr   --fr2en paraphraser.fr-en.hMoEup   --user-dir examples/translation_moe/translation_moe_src --num-experts 2  ./examples/paraphraser/searchsnippets.csv
# python examples/paraphraser/use_paraphrase_french.py   --en2fr paraphraser.en-fr   --fr2en paraphraser.fr-en.hMoEup   --user-dir examples/translation_moe/translation_moe_src --num-experts 2  ./examples/paraphraser/stackoverflow.csv
# python examples/paraphraser/use_paraphrase_french.py   --en2fr paraphraser.en-fr   --fr2en paraphraser.fr-en.hMoEup   --user-dir examples/translation_moe/translation_moe_src --num-experts 2  ./examples/paraphraser/biomedical.csv
# python examples/paraphraser/use_paraphrase_french.py   --en2fr paraphraser.en-fr   --fr2en paraphraser.fr-en.hMoEup   --user-dir examples/translation_moe/translation_moe_src --num-experts 2  ./examples/paraphraser/googlenews_TS.csv
# python examples/paraphraser/use_paraphrase_french.py   --en2fr paraphraser.en-fr   --fr2en paraphraser.fr-en.hMoEup   --user-dir examples/translation_moe/translation_moe_src --num-experts 2  ./examples/paraphraser/googlenews_T.csv
# python examples/paraphraser/use_paraphrase_french.py   --en2fr paraphraser.en-fr   --fr2en paraphraser.fr-en.hMoEup   --user-dir examples/translation_moe/translation_moe_src --num-experts 2  ./examples/paraphraser/googlenews_S.csv
# python examples/paraphraser/use_paraphrase_french.py   --en2fr paraphraser.en-fr   --fr2en paraphraser.fr-en.hMoEup   --user-dir examples/translation_moe/translation_moe_src --num-experts 2  ./examples/paraphraser/tweet.csv
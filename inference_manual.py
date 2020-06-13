import os
import torch
import json
from pprint import pprint
from argparse import ArgumentParser
from model.base import Module
from preprocess_sharc import tokenize, make_tag, convert_to_ids, MAX_LEN, compute_metrics, convert_to_ids_manual
from editor_model.base import Module as EditorModule
from preprocess_editor import trim_span


def preprocess_editor():
    data = []
    sentence = "If you’re not a UK resident, you don’t usually pay UK tax on your pension. But you might have to pay tax in the country you live in. There are a few exceptions for example, UK civil service pensions will always be taxed in the UK"
    # sentence = "[CLS] I bought nike shoes [SEP] I do not like the shoes [SEP] The delivery was bad [SEP]"
    utterance_id = 123456
    tokens, inp_ids = convert_to_ids_manual(sentence)

    for i in range(len(tokens)):
        if 'a' in tokens[i]:
            s = i
            break

    for i in range(len(tokens)):
        if 'resident' in tokens[i]:
            e = i
            break

    ex = {
        'utterance_id': utterance_id,
        'span': (s, e),
        'sentence': sentence,
        'inp': tokens,
        'type_ids': torch.tensor(torch.ones(len(tokens)), dtype=torch.long),
        'inp_ids': torch.tensor(inp_ids, dtype=torch.long),
        'inp_mask': torch.tensor(torch.ones(len(tokens)), dtype=torch.long),
    }

    data.append(ex)

    return data


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--retrieval',
                        required=True,
                        help='retrieval model to use')
    parser.add_argument('--editor', help='editor model to use (optional)')
    parser.add_argument('--fin',
                        default='sharc/json/sharc_dev.json',
                        help='input data file')
    parser.add_argument('--dout',
                        default=os.getcwd(),
                        help='directory to store output files')
    parser.add_argument('--data',
                        default='sharc/editor_disjoint',
                        help='editor data')
    parser.add_argument('--verify', action='store_true', help='run evaluation')
    parser.add_argument('--force',
                        action='store_true',
                        help='overwrite retrieval predictions')
    args = parser.parse_args()

    if not os.path.isdir(args.dout):
        os.makedirs(args.dout)

    if args.editor:
        editor_data = preprocess_editor()

        print('editor data')
        for key, val in editor_data[0].items():
            if 'scores' not in key and 'words' != key:
                pprint({key: val})

        editor = EditorModule.load(args.editor,
                                   override_args={'data': args.data})
        editor.to(editor.device)
        raw_editor_preds = editor.run_pred(editor_data)

        print('raw editor preds')
        for key, val in raw_editor_preds[0].items():
            if 'scores' not in key and 'words' != key:
                pprint({key: val})

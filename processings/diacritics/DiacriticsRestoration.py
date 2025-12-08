import os
import pickle

import numpy as np
import torch
import rb.processings.diacritics.utils as utils
from rb.core.lang import Lang
from rb.utils.downloader import check_version, download_model
from transformers import AutoTokenizer


# BertCNN hyperparameters per model size
MODEL_CONFIGS = {
    "base": {
        "window_size": 11,
        "alphabet_size": 45,
        "embedding_size": 50,
        "conv_layers": [(50, 11), (50, 2), (50, 3), (50, 4), (50, 5)],
        "fc_hidden_size": 128,
        "num_of_classes": 5,
        "batch_max_sentences": 10,
        "batch_max_windows": 280,
        "bert_model_name": "readerbench/RoBERT-base",
    },
    "small": {
        "window_size": 11,
        "alphabet_size": 45,
        "embedding_size": 50,
        "conv_layers": [(50, 11), (50, 2), (50, 3), (50, 4), (50, 5)],
        "fc_hidden_size": 128,
        "num_of_classes": 5,
        "batch_max_sentences": 10,
        "batch_max_windows": 280,
        "bert_model_name": "readerbench/RoBERT-small",
    },
}


class DiacriticsRestoration(object):
    """Diacritics restoration using BertCNN (PyTorch)."""

    def __init__(self, model_size: str = "base"):
        from rb.processings.diacritics.BertCNN import BertCNN

        if model_size not in MODEL_CONFIGS:
            raise ValueError(f"Unknown model_size '{model_size}'. Choose from: {list(MODEL_CONFIGS)}")

        cfg = MODEL_CONFIGS[model_size]
        self.max_windows = cfg["batch_max_windows"]
        self.max_sentences = cfg["batch_max_sentences"]
        self.max_sentence_length = 256

        model_dir = os.path.join("resources", "ro", "models", "diacritice", model_size)
        weights_path = os.path.join(model_dir, "weights.pt")
        char_dict_path = os.path.join(model_dir, "char_dict")

        if check_version(Lang.RO, ["models", "diacritice", model_size]):
            download_model(Lang.RO, ["models", "diacritice", model_size])

        if not os.path.exists(weights_path):
            raise FileNotFoundError(
                f"PyTorch weights not found at {weights_path}. "
                "Run scripts/convert_diacritics_weights.py to generate them."
            )

        self.model = BertCNN(
            bert_trainable=False,
            cnn_dropout_rate=0.0,
            learning_rate=0.0,
            **cfg,
        )
        state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        self.model.to(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(cfg["bert_model_name"])
        self.char_to_id_dict = pickle.load(open(char_dict_path, "rb"))

    def _run_model(self, batch):
        def to_pt(arr, dtype=torch.long):
            return torch.tensor(np.array(arr), dtype=dtype).unsqueeze(0).to(self.device)

        with torch.no_grad():
            preds, mask = self.model(
                bert_input_ids=to_pt(batch["bert_input_ids"]),
                bert_segment_ids=to_pt(batch["bert_segment_ids"]),
                token_ids=to_pt(batch["token_ids"]),
                sent_ids=to_pt(batch["sent_ids"]),
                mask=to_pt(batch["mask"], dtype=torch.float32),
                char_windows=to_pt(batch["char_windows"]),
            )
        return preds.cpu().numpy(), mask.cpu().numpy()

    def process_string(self, string, mode="replace_all"):
        full_diacritics = set("aăâiîsștț")
        explicit_diacritics = set("ăâîșțĂÂÎȘȚ")
        if len(string) > self.max_sentence_length:
            result = ""
            for i in range(0, len(string), self.max_sentence_length):
                substring = string[i:min(len(string), i + self.max_sentence_length)]
                result += self.process_string(substring, mode)
            return result

        working_string = string.lower()
        clean_string = ""
        for s in working_string:
            if s in self.char_to_id_dict.keys():
                clean_string += s
        working_string = clean_string
        working_string = ''.join([utils.get_char_basic(char) for char in working_string])

        diac_count = sum(1 for s in working_string if s in full_diacritics)

        all_predictions = []
        all_masks = []
        gen = utils.generator_bert_cnn_features_string(
            working_string, self.char_to_id_dict, 11, self.tokenizer,
            self.max_sentences, self.max_windows
        )
        steps = (diac_count // self.max_windows) + 1
        for step_idx, (batch, _) in enumerate(gen):
            if step_idx >= steps:
                break
            preds, mask = self._run_model(batch)
            all_predictions.append(preds)
            all_masks.append(mask)

        if not all_predictions:
            return string

        predictions = np.concatenate(all_predictions, axis=0)
        masks = np.concatenate(all_masks, axis=0)

        filtered_predictions = [predictions[i] for i in range(len(predictions)) if masks[i] == 1]
        predicted_classes = [np.argmax(p) for p in filtered_predictions]
        prediction_index = 0

        complete_string = ""
        for orig_char in string:
            lower_orig_char = orig_char.lower()
            if lower_orig_char in full_diacritics:
                if mode == "replace_all":
                    new_char = utils.get_char_from_label(
                        utils.get_char_basic(lower_orig_char), predicted_classes[prediction_index]
                    )
                elif mode == "replace_missing":
                    if lower_orig_char in explicit_diacritics:
                        new_char = lower_orig_char
                    else:
                        new_char = utils.get_char_from_label(
                            utils.get_char_basic(lower_orig_char), predicted_classes[prediction_index]
                        )
                prediction_index += 1
                if orig_char.isupper():
                    new_char = new_char.upper()
            else:
                new_char = orig_char
            complete_string += new_char

        return complete_string

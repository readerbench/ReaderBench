#pylint: disable=import-error
# import bert
# from bert.tokenization import FullTokenizer
import numpy as np
import sys

class BertModel_hugging:

    def __init__(self, model_dir, max_seq_len, do_lower_case=True):

        from transformers import TFAutoModel, AutoTokenizer, TFAutoModelWithLMHead
    
        if "uncased-v1" in model_dir:
            tokenizer = AutoTokenizer.from_pretrained(model_dir)
            model = TFAutoModel.from_pretrained(model_dir, from_pt=True)

        elif "tf-xlm-roberta" in model_dir:
            tokenizer = AutoTokenizer.from_pretrained(model_dir)
            model = TFAutoModel.from_pretrained(model_dir)


        self.model_dir = model_dir
        self.tokenizer = tokenizer
        
        # if "multi_cased_base" not in model_dir:
        # used to remove strip accents
        # self.tokenizer.basic_tokenizer._run_strip_accents = lambda x: x

        self.max_seq_len = max_seq_len
        # self.bert_layer = self.load_bert()
        self.bert_layer = model
        self.hidden_size = 768


    def load_bert(self):
        bert_params = bert.params_from_pretrained_ckpt(self.model_dir)
        bert_layer = bert.BertModelLayer.from_params(bert_params, name="bert_layer")
        return bert_layer

    def __get_segments(self, tokens):
        sep_enc = self.tokenizer.convert_tokens_to_ids(['[SEP]'])
        segments = []
        current_segment_id = 0
        for token in tokens:
            segments.append(current_segment_id)
            if token == sep_enc[0]:
                if current_segment_id == 0:
                    current_segment_id = 1
        return segments

    # use it with sentence2=None for single sentence
    def process_text(self, sentence1, sentence2=None):
        # sentence1 = "Tare de tot moniturul vine cu È™tand solÃ®d È™i se comportÄƒ super ok in 4ks"
        tokens = ['[CLS]']
        tokens.extend(self.tokenizer.tokenize(sentence1))
        tokens.append('[SEP]')
        if sentence2 != None:
            tokens.extend(self.tokenizer.tokenize(sentence2))
            tokens.append('[SEP]')

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_masks = [1] * len(input_ids)
        input_segments = [0] * len(input_ids)

        if self.max_seq_len == None:
            input_ids = np.array(input_ids)
            input_masks = np.array(input_masks)
            input_segments = np.array(input_segments)
            
            return input_ids, input_masks, input_segments

        else: # pad or trim
            if len(input_ids) < self.max_seq_len: # pad
                to_add = self.max_seq_len-len(input_ids)
                for _ in range(to_add):
                    input_ids.append(0)
                    input_masks.append(0)
                    input_segments.append(0)

            elif len(input_ids) > self.max_seq_len: # trim
                input_ids = input_ids[:self.max_seq_len]
                input_masks = input_masks[:self.max_seq_len]
                input_segments = input_segments[:self.max_seq_len]

            input_ids = np.array(input_ids)
            input_masks = np.array(input_masks)
            input_segments = np.array(input_segments)

            return input_ids, input_masks, input_segments

# only used for testing purposes
def test_implementation():

    import tensorflow as tf
    import tensorflow.keras as keras

    max_seq_length = 10
    bert_model = BertModel_hugging("rot/", max_seq_length)
    
    input_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32, name="input_ids")
    attn_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32, name="attn_ids")
    segment_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32, name="segment_ids")

    bert_output = bert_model.bert_layer([input_ids, attn_ids, segment_ids])[0][:,0,:]
    print(bert_output)
    print(bert_output.shape)

    model = keras.Model(inputs=[input_ids, attn_ids, segment_ids], outputs=bert_output)
    model.build(input_shape=[(None, bert_model.max_seq_len),(None, bert_model.max_seq_len), (None, bert_model.max_seq_len)])

    s1 = "dar se pare ca nu"
    # s2 = "how are you?"

    ids, attn, seg = bert_model.process_sentences(sentence1=s1, sentence2=None)
    print(ids, attn, seg)
    a = model([ids, attn, seg])
    print(a)


if __name__=="__main__":
    test_implementation()
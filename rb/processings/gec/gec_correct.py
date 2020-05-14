
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import os
from typing import Dict, List, Tuple

import numpy as np
import tensorflow as tf
from collections import namedtuple

from rb.processings.gec.transformer.transformer import Transformer
from rb.processings.gec.transformer.transformer_scheduler import CustomSchedule
from rb.processings.gec.transformer.utils import create_masks
from rb.processings.gec import beam_search
from rb.utils.downloader import check_version, download_model
from rb.core.lang import Lang
import tensorflow_datasets as tfds



class Beam(namedtuple("Beam", ["log_prob", "ids", "sentence", "length"])):
  """A finished beam

  Args:
    probs: Log probability of them beam
    ids: List of ids of the the beam
    sentence: Sentence in string
    lengths: Length of the beam
  """
  pass

class GecCorrector():

    def __init__(self, d_model: str=64, beam: int=8,
                lm_path: str=None, normalize: bool=False, weight_lm: float=1.):
        self.d_model = d_model 
        self.beam = beam 
        self.lm_path = lm_path
        self.normalize = normalize
        self.weight_lm = weight_lm
        
        if d_model == 768:
            self.checkpoint_path = os.path.join('resources', 'ro', 'models', 'gec', 'transformer_768')
            self.vocabulary_path = os.path.join(self.checkpoint_path, 'tokenizer_ro')
            if check_version(Lang.RO, ['models', 'gec', 'transformer_768']):
                download_model(Lang.RO, ['models', 'gec', 'transformer_768'])
        elif d_model == 64:
            self.checkpoint_path = os.path.join('resources', 'ro', 'models', 'gec', 'transformer_64')
            self.vocabulary_path = os.path.join(self.checkpoint_path, 'tokenizer_ro')
            if check_version(Lang.RO, ['models', 'gec', 'transformer_64']):
                download_model(Lang.RO, ['models', 'gec', 'transformer_64'])

        self.transformer, self.optimizer = self.__load_model()
        self.lm_model = self.__load_lm()
        self.tokenizer_ro = self.__load_tokenizer()
        
       

    def __load_tokenizer(self):
        tokenizer_ro = tfds.features.text.SubwordTextEncoder.load_from_file(self.vocabulary_path)
        return tokenizer_ro

    def __load_lm(self):
        if self.lm_path:
            # install kenlm from https://github.com/kpu/kenlm, but it works without it (no reranking for beams)
            import kenlm
            lm_model = kenlm.Model(lm_path)
        else:
            lm_model = None
        return lm_model
    
    def __load_model(self):
        
        self.__set_params_model(self.d_model)
        self.vocab_size = self.dict_size + 2

        learning_rate = CustomSchedule(self.d_model)
        optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, 
                                        epsilon=1e-9)

        transformer = Transformer(self.num_layers, self.d_model, self.num_heads, self.dff,
                            self.vocab_size, self.vocab_size, 
                            pe_input=self.vocab_size, 
                            pe_target=self.vocab_size,
                            rate=self.dropout)

        ckpt = tf.train.Checkpoint(transformer=transformer, optimizer=optimizer)
        
        ckpt_manager = tf.train.CheckpointManager(ckpt, self.checkpoint_path, max_to_keep=5)
        if ckpt_manager.latest_checkpoint:
            ckpt.restore(ckpt_manager.latest_checkpoint)
        else:
            tf.compat.v1.logging.error('no checkpoints for transformers, aborting')
            return None

        return transformer, optimizer

    def __set_params_model(self, d_model: int=64):
        if d_model == 64:
            self.num_layers = 2
            self.seq_length = 64
            self.dff = 64
            self.num_heads = 2
            self.dropout = 0.1
            self.dict_size = 1024

            self.num_layers = 2
        elif d_model == 768:
            self.num_layers = 6
            self.seq_length = 512
            self.dff = 2048
            self.num_heads = 8
            self.dropout = 0.1
            self.dict_size = (2**15)

    def __init_beam(self, vocab_size, end_token_id, beam_width=1):
    
        length_penalty = 0.6 if self.normalize else 0.0
        config = beam_search.BeamSearchConfig(
            beam_width=beam_width,
            vocab_size=vocab_size,
            eos_token=end_token_id,
            length_penalty_weight=length_penalty,
            choose_successors_fn=beam_search.choose_top_k)

        beam_state = beam_search.BeamSearchState(
            log_probs=tf.nn.log_softmax(tf.ones(config.beam_width)),
            lengths=tf.constant(
                1, shape=[config.beam_width], dtype=tf.int32),
            finished=tf.zeros(
                [config.beam_width], dtype=tf.bool))
        return config, beam_state

    def correct_sentence(self, inp_sentence: str): 
        start_token, end_token = [self.tokenizer_ro.vocab_size], [self.tokenizer_ro.vocab_size + 1]
        inp_sentence = inp_sentence.strip()
        in_sentence = inp_sentence
        inp_sentence = start_token + self.tokenizer_ro.encode(inp_sentence) + end_token
        if len(inp_sentence) > self.d_model:
            print('Sentence is too long (more than 766 subwords), cannot correct it.')
            return in_sentence

        start_token_id, end_token_id = self.tokenizer_ro.vocab_size, self.tokenizer_ro.vocab_size + 1

        # duplicate x beam_width == batch size
        encoder_input = tf.expand_dims(inp_sentence, 0)
        encoder_input = tf.tile(encoder_input, [self.beam, 1])

        decoder_input = [start_token_id] * self.beam
        output = tf.expand_dims(decoder_input, 1) # for batch size == beam_wisth

        # beam search init 
        config, beam_state = self.__init_beam(vocab_size=(self.dict_size + 2),
                                                    end_token_id=end_token_id, 
                                                    beam_width=self.beam)
        beam_values = tf.constant(start_token_id, shape=(1, self.beam))
        beam_parents = tf.zeros((2, self.beam), dtype=tf.int32)
        
        for i in range(self.d_model + 2):
            enc_padding_mask, combined_mask, dec_padding_mask = create_masks(
                encoder_input, output)

            predictions, attention_weights = self.transformer(encoder_input, output,
                                                                False, enc_padding_mask,
                                                                combined_mask, dec_padding_mask)
            # !predictions.shape == (batch_size, i, vocab_size) (predicts a softmax for each existing word)
            beam_pred = tf.squeeze(predictions[: ,-1:, :], 1)  # (batch_size, 1, vocab_size), select only the last word
            bs_output, beam_state = beam_search.beam_search_step(time_=i, logits=beam_pred,
                                                                beam_state=beam_state, config=config)

            # add new predictions to the beams decoder
            bs_output_predicted_ids = tf.expand_dims(bs_output.predicted_ids, axis=0)
            beam_values = tf.concat([beam_values, bs_output_predicted_ids], axis=0)
            res = tf.cast(beam_search.gather_tree_py(beam_values.numpy(), beam_parents.numpy()), dtype=tf.int32)
            output = tf.transpose(res)

            bs_output_beam_parent_ids = tf.expand_dims(bs_output.beam_parent_ids, axis=0)
            beam_parents = tf.concat([beam_parents, bs_output_beam_parent_ids], axis=0)

            all_finished = tf.reduce_all(beam_state.finished) # and
            if all_finished:    break

        beams = []
        for i, out in enumerate(output):
            b = Beam(log_prob=beam_state.log_probs[i].numpy(), sentence="", ids=out.numpy(), length=len(out.numpy()))
            beams.append(b)

        sentences = self.__decode_sentence(beams)
        # for sent in sentences:
        #     print(sent.log_prob, sent.sentence)
        sentences = sorted(sentences, key=lambda beam: beam.log_prob, reverse=True)
        return sentences[0].sentence
    
    def __decode_sentence(self, beams):
        sentences = []

        for beam in beams:
            sentence_ids = []

            for i in beam.ids:
                if i < self.tokenizer_ro.vocab_size:
                    sentence_ids.append(i)
                if i == self.tokenizer_ro.vocab_size + 1:
                    break
                
            predicted_sentence = self.tokenizer_ro.decode(sentence_ids)

            if self.lm_model:
                lm_prob = lm_model.score(predicted_sentence, bos=True, eos=True)

                if self.normalize:
                    cand_prob = beam.log_prob + self.weight_lm * lm_prob * (1.0/beam.length)
                else:
                    cand_prob = beam.log_prob + self.weight_lm * lm_prob
            else:
                cand_prob = beam.log_prob

            sentences.append(Beam(log_prob=cand_prob, sentence=predicted_sentence,
                                 ids=beam.ids, length=len(beam.ids)))
        return sentences


if __name__ == "__main__":
    corrector = GecCorrector(checkpoint_path='checkpoints/10m_transformer_768_2',
                vocabulary_path='checkpoints/10m_transformer_768_2/tokenizer_ro', d_model=768)

    decoded_senntence = corrector.correct_sentence('Am mers la magazi sa cuumper leegume si fructe de mancaat.')
    print(decoded_senntence)
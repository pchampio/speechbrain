"""
Token scorer abstraction and specifications.

Authors:
 * Adel Moumen 2022, 2023
 * Sung-Lin Yeh 2021
"""

import torch
import numpy as np
import speechbrain as sb
from speechbrain.decoders.ctc import CTCPrefixScore


class BaseScorerInterface:
    """A scorer abstraction to be inherited by other
    scoring approaches for beam search.

    A scorer is a module that scores tokens in vocabulary
    based on the current timestep input and the previous
    scorer states. It can be used to score on full vocabulary
    set (i.e., full scorers) or a pruned set of tokens (i.e. partial scorers)
    to prevent computation overhead. In the latter case, the partial scorers
    will be called after the full scorers. It will only scores the
    top-k candidates (i.e., pruned set of tokens) extracted from the full scorers.
    The top-k candidates are extracted based on the beam size and the
    scorer_beam_scale such that the number of candidates is
    int(beam_size * scorer_beam_scale). It can be very useful
    when the full scorers are computationally expensive (e.g., KenLM scorer).

    Inherit this class to implement your own scorer compatible with
    speechbrain.decoders.seq2seq.S2SBeamSearcher().

    See:
        - speechbrain.decoders.scorer.CTCPrefixScorer
        - speechbrain.decoders.scorer.RNNLMScorer
        - speechbrain.decoders.scorer.TransformerLMScorer
        - speechbrain.decoders.scorer.KenLMScorer
        - speechbrain.decoders.scorer.CoverageScorer
        - speechbrain.decoders.scorer.LengthScorer
    """

    def score(self, inp_tokens, memory, candidates, attn):
        """This method scores the new beams based on the
        informations of the current timestep.

        A score is a tensor of shape (batch_size x beam_size, vocab_size).
        It is the log probability of the next token given the current
        timestep input and the previous scorer states.

        It can be used to score on pruned top-k candidates
        to prevent computation overhead, or on full vocabulary set
        when candidates is None.

        Arguments
        ---------
        inp_tokens : torch.Tensor
            The input tensor of the current timestep.
        memory : No limit
            The scorer states for this timestep.
        candidates : torch.Tensor
            (batch_size x beam_size, scorer_beam_size).
            The top-k candidates to be scored after the full scorers.
            If None, scorers will score on full vocabulary set.
        attn : torch.Tensor
            The attention weight to be used in CoverageScorer or CTCScorer.

        Returns
        ---------
        torch.Tensor
            (batch_size x beam_size, vocab_size), Scores for the next tokens.
        memory : No limit
            The memory variables input for this timestep.
        """
        raise NotImplementedError

    def permute_mem(self, memory, index):
        """This method permutes the scorer memory to synchronize
        the memory index with the current output and perform
        batched beam search.

        Arguments
        ---------
        memory : No limit
            The memory variables input for this timestep.
        index : torch.Tensor
            (batch_size, beam_size). The index of the previous path.
        """
        return None

    def reset_mem(self, x, enc_lens):
        """This method should implement the resetting of
        memory variables for the scorer.

        Arguments
        ---------
        x : torch.Tensor
            The precomputed encoder states to be used when decoding.
            (ex. the encoded speech representation to be attended).
        enc_lens : torch.Tensor
            The speechbrain-style relative length.
        """
        return None


class CTCScorer(BaseScorerInterface):
    """A wrapper of CTCPrefixScore based on the BaseScorerInterface.

    This Scorer is used to provides the CTC label-synchronous scores
    of the next input tokens. The implementation is based on
    https://www.merl.com/publications/docs/TR2017-190.pdf.

    See:
        - speechbrain.decoders.scorer.CTCPrefixScore

    Arguments
    ---------
    ctc_fc : torch.nn.Module
        A output linear layer for ctc.
    blank_index : int
        The index of the blank token.
    eos_index : int
        The index of the end-of-sequence (eos) token.
    ctc_window_size : int
        Compute the ctc scores over the time frames using windowing
        based on attention peaks. If 0, no windowing applied. (default: 0)

    Example
    -------
    >>> import torch
    >>> from speechbrain.nnet.linear import Linear
    >>> from speechbrain.lobes.models.transformer.TransformerASR import TransformerASR
    >>> from speechbrain.decoders import S2STransformerBeamSearcher, CTCScorer, ScorerBuilder
    >>> batch_size=8
    >>> n_channels=6
    >>> input_size=40
    >>> d_model=128
    >>> tgt_vocab=140
    >>> src = torch.rand([batch_size, n_channels, input_size])
    >>> tgt = torch.randint(0, tgt_vocab, [batch_size, n_channels])
    >>> net = TransformerASR(
    ...    tgt_vocab, input_size, d_model, 8, 1, 1, 1024, activation=torch.nn.GELU
    ... )
    >>> ctc_lin = Linear(input_shape=(1, 40, d_model), n_neurons=tgt_vocab)
    >>> lin = Linear(input_shape=(1, 40, d_model), n_neurons=tgt_vocab)
    >>> eos_index = 2
    >>> ctc_scorer = CTCScorer(
    ...    ctc_fc=ctc_lin,
    ...    blank_index=0,
    ...    eos_index=eos_index,
    ... )
    >>> scorer = ScorerBuilder(
    ...     full_scorers=[ctc_scorer],
    ...     weights={'ctc': 1.0}
    ... )
    >>> searcher = S2STransformerBeamSearcher(
    ...     modules=[net, lin],
    ...     bos_index=1,
    ...     eos_index=eos_index,
    ...     min_decode_ratio=0.0,
    ...     max_decode_ratio=1.0,
    ...     using_eos_threshold=False,
    ...     beam_size=7,
    ...     temperature=1.15,
    ...     scorer=scorer
    ... )
    >>> enc, dec = net.forward(src, tgt)
    >>> hyps, _, _, _ = searcher(enc, torch.ones(batch_size))
    """

    def __init__(
        self, ctc_fc, blank_index, eos_index, ctc_window_size=0,
    ):
        self.ctc_fc = ctc_fc
        self.blank_index = blank_index
        self.eos_index = eos_index
        self.ctc_window_size = ctc_window_size
        self.softmax = sb.nnet.activations.Softmax(apply_log=True)

    def score(self, inp_tokens, memory, candidates, attn):
        """This method scores the new beams based on the
        CTC scores computed over the time frames.

        See:
            - speechbrain.decoders.scorer.CTCPrefixScore

        Arguments
        ---------
        inp_tokens : torch.Tensor
            The input tensor of the current timestep.
        memory : No limit
            The scorer states for this timestep.
        candidates : torch.Tensor
            (batch_size x beam_size, scorer_beam_size).
            The top-k candidates to be scored after the full scorers.
            If None, scorers will score on full vocabulary set.
        attn : torch.Tensor
            The attention weight to be used in CoverageScorer or CTCScorer.
        """
        scores, memory = self.ctc_score.forward_step(
            inp_tokens, memory, candidates, attn
        )
        return scores, memory

    def permute_mem(self, memory, index):
        """This method permutes the scorer memory to synchronize
        the memory index with the current output and perform
        batched CTC beam search.

        Arguments
        ---------
        memory : No limit
            The memory variables input for this timestep.
        index : torch.Tensor
            (batch_size, beam_size). The index of the previous path.
        """
        r, psi = self.ctc_score.permute_mem(memory, index)
        return r, psi

    def reset_mem(self, x, enc_lens):
        """This method implement the resetting of
        memory variables for the CTC scorer.

        Arguments
        ---------
        x : torch.Tensor
            The precomputed encoder states to be used when decoding.
            (ex. the encoded speech representation to be attended).
        enc_lens : torch.Tensor
            The speechbrain-style relative length.
        """
        logits = self.ctc_fc(x)
        x = self.softmax(logits)
        self.ctc_score = CTCPrefixScore(
            x, enc_lens, self.blank_index, self.eos_index, self.ctc_window_size
        )
        return None


class RNNLMScorer(BaseScorerInterface):
    """A wrapper of RNNLM based on BaseScorerInterface.

    The RNNLMScorer is used to provide the RNNLM scores of the next input tokens
    based on the current timestep input and the previous scorer states.

    Arguments
    ---------
    language_model : torch.nn.Module
        A RNN-based language model.
    temperature : float
        Temperature factor applied to softmax. It changes the probability
        distribution, being softer when T>1 and sharper with T<1. (default: 1.0)

    Example
    -------
    >>> from speechbrain.nnet.linear import Linear
    >>> from speechbrain.lobes.models.RNNLM import RNNLM
    >>> from speechbrain.nnet.RNN import AttentionalRNNDecoder
    >>> from speechbrain.decoders import S2SRNNBeamSearcher, RNNLMScorer, ScorerBuilder
    >>> input_size=17
    >>> vocab_size=11
    >>> emb = torch.nn.Embedding(
    ...     embedding_dim=input_size,
    ...     num_embeddings=vocab_size,
    ... )
    >>> d_model=7
    >>> dec = AttentionalRNNDecoder(
    ...     rnn_type="gru",
    ...     attn_type="content",
    ...     hidden_size=3,
    ...     attn_dim=3,
    ...     num_layers=1,
    ...     enc_dim=d_model,
    ...     input_size=input_size,
    ... )
    >>> n_channels=3
    >>> seq_lin = Linear(input_shape=[d_model, n_channels], n_neurons=vocab_size)
    >>> lm_weight = 0.4
    >>> lm_model = RNNLM(
    ...     embedding_dim=d_model,
    ...     output_neurons=vocab_size,
    ...     dropout=0.0,
    ...     rnn_neurons=128,
    ...     dnn_neurons=64,
    ...     return_hidden=True,
    ... )
    >>> rnnlm_scorer = RNNLMScorer(
    ...     language_model=lm_model,
    ...     temperature=1.25,
    ... )
    >>> scorer = ScorerBuilder(
    ...     full_scorers=[rnnlm_scorer],
    ...     weights={'rnnlm': lm_weight}
    ... )
    >>> beam_size=5
    >>> searcher = S2SRNNBeamSearcher(
    ...     embedding=emb,
    ...     decoder=dec,
    ...     linear=seq_lin,
    ...     bos_index=1,
    ...     eos_index=2,
    ...     min_decode_ratio=0.0,
    ...     max_decode_ratio=1.0,
    ...     topk=2,
    ...     using_eos_threshold=False,
    ...     beam_size=beam_size,
    ...     temperature=1.25,
    ...     scorer=scorer
    ... )
    >>> batch_size=2
    >>> enc = torch.rand([batch_size, n_channels, d_model])
    >>> wav_len = torch.ones([batch_size])
    >>> hyps, _, _, _ = searcher(enc, wav_len)
    """

    def __init__(self, language_model, temperature=1.0):
        self.lm = language_model
        self.lm.eval()
        self.temperature = temperature
        self.softmax = sb.nnet.activations.Softmax(apply_log=True)

    def score(self, inp_tokens, memory, candidates, attn):
        """This method scores the new beams based on the
        RNNLM scores computed over the previous tokens.

        Arguments
        ---------
        inp_tokens : torch.Tensor
            The input tensor of the current timestep.
        memory : No limit
            The scorer states for this timestep.
        candidates : torch.Tensor
            (batch_size x beam_size, scorer_beam_size).
            The top-k candidates to be scored after the full scorers.
            If None, scorers will score on full vocabulary set.
        attn : torch.Tensor
            The attention weight to be used in CoverageScorer or CTCScorer.
        """
        with torch.no_grad():
            logits, hs = self.lm(inp_tokens, hx=memory)
            log_probs = self.softmax(logits / self.temperature)
        return log_probs, hs

    def permute_mem(self, memory, index):
        """This method permutes the scorer memory to synchronize
        the memory index with the current output and perform
        batched beam search.

        Arguments
        ---------
        memory : No limit
            The memory variables input for this timestep.
        index : torch.Tensor
            (batch_size, beam_size). The index of the previous path.
        """
        if isinstance(memory, tuple):
            memory_0 = torch.index_select(memory[0], dim=1, index=index)
            memory_1 = torch.index_select(memory[1], dim=1, index=index)
            memory = (memory_0, memory_1)
        else:
            memory = torch.index_select(memory, dim=1, index=index)
        return memory

    def reset_mem(self, x, enc_lens):
        """This method implement the resetting of
        memory variables for the RNNLM scorer.

        Arguments
        ---------
        x : torch.Tensor
            The precomputed encoder states to be used when decoding.
            (ex. the encoded speech representation to be attended).
        enc_lens : torch.Tensor
            The speechbrain-style relative length.
        """
        return None


class TransformerLMScorer(BaseScorerInterface):
    """A wrapper of TransformerLM based on BaseScorerInterface.

    The TransformerLMScorer is used to provide the TransformerLM scores
    of the next input tokens based on the current timestep input and the
    previous scorer states.

    Arguments
    ---------
    language_model : torch.nn.Module
        A Transformer-based language model.
    temperature : float
        Temperature factor applied to softmax. It changes the probability
        distribution, being softer when T>1 and sharper with T<1. (default: 1.0)

    Example
    -------
    >>> from speechbrain.nnet.linear import Linear
    >>> from speechbrain.lobes.models.transformer.TransformerASR import TransformerASR
    >>> from speechbrain.lobes.models.transformer.TransformerLM import TransformerLM
    >>> from speechbrain.decoders import S2STransformerBeamSearcher, TransformerLMScorer, CTCScorer, ScorerBuilder
    >>> input_size=17
    >>> vocab_size=11
    >>> d_model=128
    >>> net = TransformerASR(
    ...     tgt_vocab=vocab_size,
    ...     input_size=input_size,
    ...     d_model=d_model,
    ...     nhead=8,
    ...     num_encoder_layers=1,
    ...     num_decoder_layers=1,
    ...     d_ffn=256,
    ...     activation=torch.nn.GELU
    ... )
    >>> lm_model = TransformerLM(
    ...     vocab=vocab_size,
    ...     d_model=d_model,
    ...     nhead=8,
    ...     num_encoder_layers=1,
    ...     num_decoder_layers=0,
    ...     d_ffn=256,
    ...     activation=torch.nn.GELU,
    ... )
    >>> n_channels=6
    >>> ctc_lin = Linear(input_size=d_model, n_neurons=vocab_size)
    >>> seq_lin = Linear(input_size=d_model, n_neurons=vocab_size)
    >>> eos_index = 2
    >>> ctc_scorer = CTCScorer(
    ...     ctc_fc=ctc_lin,
    ...     blank_index=0,
    ...     eos_index=eos_index,
    ... )
    >>> transformerlm_scorer = TransformerLMScorer(
    ...     language_model=lm_model,
    ...     temperature=1.15,
    ... )
    >>> ctc_weight_decode=0.4
    >>> lm_weight=0.6
    >>> scorer = ScorerBuilder(
    ...     full_scorers=[transformerlm_scorer, ctc_scorer],
    ...     weights={'transformerlm': lm_weight, 'ctc': ctc_weight_decode}
    ... )
    >>> beam_size=5
    >>> searcher = S2STransformerBeamSearcher(
    ...     modules=[net, seq_lin],
    ...     bos_index=1,
    ...     eos_index=eos_index,
    ...     min_decode_ratio=0.0,
    ...     max_decode_ratio=1.0,
    ...     using_eos_threshold=False,
    ...     beam_size=beam_size,
    ...     temperature=1.15,
    ...     scorer=scorer
    ... )
    >>> batch_size=2
    >>> wav_len = torch.ones([batch_size])
    >>> src = torch.rand([batch_size, n_channels, input_size])
    >>> tgt = torch.randint(0, vocab_size, [batch_size, n_channels])
    >>> enc, dec = net.forward(src, tgt)
    >>> hyps, _, _, _ = searcher(enc, wav_len)
    """

    def __init__(self, language_model, temperature=1.0):
        self.lm = language_model
        self.lm.eval()
        self.temperature = temperature
        self.softmax = sb.nnet.activations.Softmax(apply_log=True)

    def score(self, inp_tokens, memory, candidates, attn):
        """This method scores the new beams based on the
        TransformerLM scores computed over the previous tokens.

        Arguments
        ---------
        inp_tokens : torch.Tensor
            The input tensor of the current timestep.
        memory : No limit
            The scorer states for this timestep.
        candidates : torch.Tensor
            (batch_size x beam_size, scorer_beam_size).
            The top-k candidates to be scored after the full scorers.
            If None, scorers will score on full vocabulary set.
        attn : torch.Tensor
            The attention weight to be used in CoverageScorer or CTCScorer.
        """
        with torch.no_grad():
            if memory is None:
                memory = torch.empty(
                    inp_tokens.size(0), 0, device=inp_tokens.device
                )
            # Append the predicted token of the previous step to existing memory.
            memory = torch.cat([memory, inp_tokens.unsqueeze(1)], dim=-1)
            if not next(self.lm.parameters()).is_cuda:
                self.lm.to(inp_tokens.device)
            logits = self.lm(memory)
            log_probs = self.softmax(logits / self.temperature)
        return log_probs[:, -1, :], memory

    def permute_mem(self, memory, index):
        """This method permutes the scorer memory to synchronize
        the memory index with the current output and perform
        batched beam search.

        Arguments
        ---------
        memory : No limit
            The memory variables input for this timestep.
        index : torch.Tensor
            (batch_size, beam_size). The index of the previous path.
        """
        memory = torch.index_select(memory, dim=0, index=index)
        return memory

    def reset_mem(self, x, enc_lens):
        """This method implement the resetting of
        memory variables for the RNNLM scorer.

        Arguments
        ---------
        x : torch.Tensor
            The precomputed encoder states to be used when decoding.
            (ex. the encoded speech representation to be attended).
        enc_lens : torch.Tensor
            The speechbrain-style relative length.
        """
        return None


class KenLMScorer(BaseScorerInterface):
    """KenLM N-gram scorer.

    This scorer is based on KenLM, which is a fast and efficient
    N-gram language model toolkit. It is used to provide the n-gram scores
    of the next input tokens.

    This scorer is dependent on the KenLM package. It can be installed
    with the following command:
            > pip install https://github.com/kpu/kenlm/archive/master.zip

    Note: The KenLM scorer is computationally expensive. It is recommended
    to use it as a partial scorer to score on the top-k candidates instead
    of the full vocabulary set.

    Arguments
    ---------
    lm_path : str
        The path of ngram model.
    vocab_size: int
        The total number of tokens.
    token_list : list
        The tokens set.

    # Example
    # -------
    # >>> from speechbrain.nnet.linear import Linear
    # >>> from speechbrain.nnet.RNN import AttentionalRNNDecoder
    # >>> from speechbrain.decoders import S2SRNNBeamSearcher, KenLMScorer, ScorerBuilder
    # >>> input_size=17
    # >>> vocab_size=11
    # >>> lm_path='path/to/kenlm_model.arpa' # or .bin
    # >>> token_list=['<pad>', '<bos>', '<eos>', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']
    # >>> emb = torch.nn.Embedding(
    # ...     embedding_dim=input_size,
    # ...     num_embeddings=vocab_size,
    # ... )
    # >>> d_model=7
    # >>> dec = AttentionalRNNDecoder(
    # ...     rnn_type="gru",
    # ...     attn_type="content",
    # ...     hidden_size=3,
    # ...     attn_dim=3,
    # ...     num_layers=1,
    # ...     enc_dim=d_model,
    # ...     input_size=input_size,
    # ... )
    # >>> n_channels=3
    # >>> seq_lin = Linear(input_shape=[d_model, n_channels], n_neurons=vocab_size)
    # >>> kenlm_weight = 0.4
    # >>> kenlm_model = KenLMScorer(
    # ...     lm_path=lm_path,
    # ...     vocab_size=vocab_size,
    # ...     token_list=token_list,
    # ... )
    # >>> scorer = ScorerBuilder(
    # ...     full_scorers=[kenlm_model],
    # ...     weights={'kenlm': kenlm_weight}
    # ... )
    # >>> beam_size=5
    # >>> searcher = S2SRNNBeamSearcher(
    # ...     embedding=emb,
    # ...     decoder=dec,
    # ...     linear=seq_lin,
    # ...     bos_index=1,
    # ...     eos_index=2,
    # ...     min_decode_ratio=0.0,
    # ...     max_decode_ratio=1.0,
    # ...     topk=2,
    # ...     using_eos_threshold=False,
    # ...     beam_size=beam_size,
    # ...     temperature=1.25,
    # ...     scorer=scorer
    # ... )
    # >>> batch_size=2
    # >>> enc = torch.rand([batch_size, n_channels, d_model])
    # >>> wav_len = torch.ones([batch_size])
    # >>> hyps, _, _, _ = searcher(enc, wav_len)
    """

    def __init__(self, lm_path, vocab_size, token_list):
        try:
            import kenlm

            self.kenlm = kenlm
        except ImportError:
            MSG = """Couldn't import KenLM
            It is an optional dependency; it is not installed with SpeechBrain
            by default. Install it with:
            > pip install https://github.com/kpu/kenlm/archive/master.zip
            """
            raise ImportError(MSG)
        self.lm = self.kenlm.Model(lm_path)
        self.vocab_size = vocab_size
        self.full_candidates = np.arange(self.vocab_size)
        self.minus_inf = -1e20
        if len(token_list) != vocab_size:
            MSG = "The size of the token_list and vocab_size are not matched."
            raise ValueError(MSG)
        self.id2char = token_list

    def score(self, inp_tokens, memory, candidates, attn):
        """This method scores the new beams based on the
        n-gram scores.

        Arguments
        ---------
        inp_tokens : torch.Tensor
            The input tensor of the current timestep.
        memory : No limit
            The scorer states for this timestep.
        candidates : torch.Tensor
            (batch_size x beam_size, scorer_beam_size).
            The top-k candidates to be scored after the full scorers.
            If None, scorers will score on full vocabulary set.
        attn : torch.Tensor
            The attention weight to be used in CoverageScorer or CTCScorer.
        """
        n_bh = inp_tokens.size(0)
        scale = 1.0 / np.log10(np.e)

        if memory is None:
            state = self.kenlm.State()
            state = np.array([state] * n_bh)
            scoring_table = np.ones(n_bh)
        else:
            state, scoring_table = memory

        # Perform full scorer mode, not recommend
        if candidates is None:
            candidates = [self.full_candidates] * n_bh

        # Store new states and scores
        scores = np.ones((n_bh, self.vocab_size)) * self.minus_inf
        new_memory = np.zeros((n_bh, self.vocab_size), dtype=object)
        new_scoring_table = np.ones((n_bh, self.vocab_size)) * -1
        # Scoring
        for i in range(n_bh):
            if scoring_table[i] == -1:
                continue
            parent_state = state[i]
            for token_id in candidates[i]:
                char = self.id2char[token_id.item()]
                out_state = self.kenlm.State()
                score = scale * self.lm.BaseScore(parent_state, char, out_state)
                scores[i, token_id] = score
                new_memory[i, token_id] = out_state
                new_scoring_table[i, token_id] = 1
        scores = torch.from_numpy(scores).float().to(inp_tokens.device)
        return scores, (new_memory, new_scoring_table)

    def permute_mem(self, memory, index):
        """This method permutes the scorer memory to synchronize
        the memory index with the current output and perform
        batched beam search.

        Arguments
        ---------
        memory : No limit
            The memory variables input for this timestep.
        index : torch.Tensor
            (batch_size, beam_size). The index of the previous path.
        """
        state, scoring_table = memory

        index = index.cpu().numpy()
        # The first index of each sentence.
        beam_size = index.shape[1]
        beam_offset = self.batch_index * beam_size
        hyp_index = (
            index
            + np.broadcast_to(np.expand_dims(beam_offset, 1), index.shape)
            * self.vocab_size
        )
        hyp_index = hyp_index.reshape(-1)
        # Update states
        state = state.reshape(-1)
        state = state[hyp_index]
        scoring_table = scoring_table.reshape(-1)
        scoring_table = scoring_table[hyp_index]
        return state, scoring_table

    def reset_mem(self, x, enc_lens):
        """This method implement the resetting of
        memory variables for the KenLM scorer.

        Arguments
        ---------
        x : torch.Tensor
            The precomputed encoder states to be used when decoding.
            (ex. the encoded speech representation to be attended).
        enc_lens : torch.Tensor
            The speechbrain-style relative length.
        """
        state = self.kenlm.State()
        self.lm.NullContextWrite(state)
        self.batch_index = np.arange(x.size(0))
        return None


class CoverageScorer(BaseScorerInterface):
    """A coverage penalty scorer to prevent looping of hyps,
    where ```coverage``` is the cumulative attention probability vector.
    Reference: https://arxiv.org/pdf/1612.02695.pdf,
               https://arxiv.org/pdf/1808.10792.pdf

    Arguments
    ---------
    vocab_size: int
        The total number of tokens.
    threshold: float
        The penalty increases when the coverage of a frame is more
        than given threshold. (default: 0.5)

    Example
    -------
    >>> from speechbrain.nnet.linear import Linear
    >>> from speechbrain.lobes.models.RNNLM import RNNLM
    >>> from speechbrain.nnet.RNN import AttentionalRNNDecoder
    >>> from speechbrain.decoders import S2SRNNBeamSearcher, RNNLMScorer, CoverageScorer, ScorerBuilder
    >>> input_size=17
    >>> vocab_size=11
    >>> emb = torch.nn.Embedding(
    ...     num_embeddings=vocab_size,
    ...     embedding_dim=input_size
    ... )
    >>> d_model=7
    >>> dec = AttentionalRNNDecoder(
    ...     rnn_type="gru",
    ...     attn_type="content",
    ...     hidden_size=3,
    ...     attn_dim=3,
    ...     num_layers=1,
    ...     enc_dim=d_model,
    ...     input_size=input_size,
    ... )
    >>> n_channels=3
    >>> seq_lin = Linear(input_shape=[d_model, n_channels], n_neurons=vocab_size)
    >>> lm_weight = 0.4
    >>> coverage_penalty = 1.0
    >>> lm_model = RNNLM(
    ...     embedding_dim=d_model,
    ...     output_neurons=vocab_size,
    ...     dropout=0.0,
    ...     rnn_neurons=128,
    ...     dnn_neurons=64,
    ...     return_hidden=True,
    ... )
    >>> rnnlm_scorer = RNNLMScorer(
    ...     language_model=lm_model,
    ...     temperature=1.25,
    ... )
    >>> coverage_scorer = CoverageScorer(vocab_size=vocab_size)
    >>> scorer = ScorerBuilder(
    ...     full_scorers=[rnnlm_scorer, coverage_scorer],
    ...     weights={'rnnlm': lm_weight, 'coverage': coverage_penalty}
    ... )
    >>> beam_size=5
    >>> searcher = S2SRNNBeamSearcher(
    ...     embedding=emb,
    ...     decoder=dec,
    ...     linear=seq_lin,
    ...     bos_index=1,
    ...     eos_index=2,
    ...     min_decode_ratio=0.0,
    ...     max_decode_ratio=1.0,
    ...     topk=2,
    ...     using_eos_threshold=False,
    ...     beam_size=beam_size,
    ...     temperature=1.25,
    ...     scorer=scorer
    ... )
    >>> batch_size=2
    >>> enc = torch.rand([batch_size, n_channels, d_model])
    >>> wav_len = torch.ones([batch_size])
    >>> hyps, _, _, _ = searcher(enc, wav_len)
    """

    def __init__(self, vocab_size, threshold=0.5):
        self.vocab_size = vocab_size
        self.threshold = threshold
        # Use time_step to normalize the coverage over steps
        self.time_step = 0

    def score(self, inp_tokens, coverage, candidates, attn):
        """Specifies token scoring."""
        n_bh = attn.size(0)
        self.time_step += 1

        if coverage is None:
            coverage = torch.zeros_like(attn, device=attn.device)

        # Current coverage
        if len(attn.size()) > 2:
            # the attn of transformer is [batch_size x beam_size, current_step, source_len]
            coverage = torch.sum(attn, dim=1)
        else:
            coverage = coverage + attn

        # Compute coverage penalty and add it to scores
        penalty = torch.max(
            coverage, coverage.clone().fill_(self.threshold)
        ).sum(-1)
        penalty = penalty - coverage.size(-1) * self.threshold
        penalty = penalty.view(n_bh).unsqueeze(1).expand(-1, self.vocab_size)
        return -1 * penalty / self.time_step, coverage

    def permute_mem(self, coverage, index):
        """Specifies memory synchronisation."""
        # Update coverage
        coverage = torch.index_select(coverage, dim=0, index=index)
        return coverage

    def reset_mem(self, x, enc_lens):
        """Specifies memory reset."""
        self.time_step = 0
        return None


class LengthScorer(BaseScorerInterface):
    """A length rewarding scorer.

    Arguments
    ---------
    vocab_size: int
        The total number of tokens.
    """

    def __init__(self, vocab_size):
        self.vocab_size = vocab_size

    def score(self, inp_tokens, memory, candidates, attn):
        """Specifies token scoring."""
        return (
            torch.tensor(
                [1.0], device=inp_tokens.device, dtype=inp_tokens.dtype
            ).expand(inp_tokens.size(0), self.vocab_size),
            None,
        )


class ScorerBuilder:
    """ Builds scorer instance for beamsearch.

    The ScorerBuilder class is responsible for building a scorer instance for
    beam search. It takes weights for full and partial scorers, as well as
    instances of full and partial scorer classes. It combines the scorers based
    on the weights specified and provides methods for scoring tokens, permuting
    scorer memory, and resetting scorer memory.

    This is the class to be used for building scorer instances for beam search.

    See speechbrain.decoders.seq2seq.S2SBeamSearcher()

    Arguments
    ---------
    weights : dict
        Weights of full/partial scorers specified.
    full_scorers : list
        Scorers that score on full vocabulary set.
    partial_scorers : list
        Scorers that score on pruned tokens to prevent computation overhead.
        Partial scoring is performed after full scorers.
    scorer_beam_scale : float
        The scale decides the number of pruned tokens for partial scorers:
        int(beam_size * scorer_beam_scale).

    Example
    -------
    >>> from speechbrain.nnet.linear import Linear
    >>> from speechbrain.lobes.models.transformer.TransformerASR import TransformerASR
    >>> from speechbrain.lobes.models.transformer.TransformerLM import TransformerLM
    >>> from speechbrain.decoders import S2STransformerBeamSearcher, TransformerLMScorer, CoverageScorer, CTCScorer, ScorerBuilder
    >>> input_size=17
    >>> vocab_size=11
    >>> d_model=128
    >>> net = TransformerASR(
    ...     tgt_vocab=vocab_size,
    ...     input_size=input_size,
    ...     d_model=d_model,
    ...     nhead=8,
    ...     num_encoder_layers=1,
    ...     num_decoder_layers=1,
    ...     d_ffn=256,
    ...     activation=torch.nn.GELU
    ... )
    >>> lm_model = TransformerLM(
    ...     vocab=vocab_size,
    ...     d_model=d_model,
    ...     nhead=8,
    ...     num_encoder_layers=1,
    ...     num_decoder_layers=0,
    ...     d_ffn=256,
    ...     activation=torch.nn.GELU,
    ... )
    >>> n_channels=6
    >>> ctc_lin = Linear(input_size=d_model, n_neurons=vocab_size)
    >>> seq_lin = Linear(input_size=d_model, n_neurons=vocab_size)
    >>> eos_index = 2
    >>> ctc_scorer = CTCScorer(
    ...     ctc_fc=ctc_lin,
    ...     blank_index=0,
    ...     eos_index=eos_index,
    ... )
    >>> transformerlm_scorer = TransformerLMScorer(
    ...     language_model=lm_model,
    ...     temperature=1.15,
    ... )
    >>> coverage_scorer = CoverageScorer(vocab_size=vocab_size)
    >>> ctc_weight_decode=0.4
    >>> lm_weight=0.6
    >>> coverage_penalty = 1.0
    >>> scorer = ScorerBuilder(
    ...     full_scorers=[transformerlm_scorer, coverage_scorer],
    ...     partial_scorers=[ctc_scorer],
    ...     weights={'transformerlm': lm_weight, 'ctc': ctc_weight_decode, 'coverage': coverage_penalty}
    ... )
    >>> beam_size=5
    >>> searcher = S2STransformerBeamSearcher(
    ...     modules=[net, seq_lin],
    ...     bos_index=1,
    ...     eos_index=eos_index,
    ...     min_decode_ratio=0.0,
    ...     max_decode_ratio=1.0,
    ...     using_eos_threshold=False,
    ...     beam_size=beam_size,
    ...     topk=3,
    ...     temperature=1.15,
    ...     scorer=scorer
    ... )
    >>> batch_size=2
    >>> wav_len = torch.ones([batch_size])
    >>> src = torch.rand([batch_size, n_channels, input_size])
    >>> tgt = torch.randint(0, vocab_size, [batch_size, n_channels])
    >>> enc, dec = net.forward(src, tgt)
    >>> hyps, _, _, _  = searcher(enc, wav_len)
    """

    def __init__(
        self,
        weights=dict(),
        full_scorers=list(),
        partial_scorers=list(),
        scorer_beam_scale=1.5,
    ):
        assert len(weights) == len(full_scorers) + len(
            partial_scorers
        ), "Weights and scorers are not matched."

        self.scorer_beam_scale = scorer_beam_scale
        all_scorer_names = [
            k.lower().split("scorer")[0]
            for k in globals().keys()
            if k.endswith("Scorer")
        ]
        full_scorer_names = [
            impl.__class__.__name__.lower().split("scorer")[0]
            for impl in full_scorers
        ]
        partial_scorer_names = [
            impl.__class__.__name__.lower().split("scorer")[0]
            for impl in partial_scorers
        ]

        # Have a default 0.0 weight for scorer not specified
        init_weights = {k: 0.0 for k in all_scorer_names}
        self.weights = {**init_weights, **weights}
        self.full_scorers = dict(zip(full_scorer_names, full_scorers))
        self.partial_scorers = dict(zip(partial_scorer_names, partial_scorers))

        # Check if scorers are valid
        self._validate_scorer(all_scorer_names)

    def score(self, inp_tokens, memory, attn, log_probs, beam_size):
        """This method scores tokens in vocabulary based on defined full scorers
        and partial scorers. Scores will be added to the log probs for beamsearch.

        Arguments
        ---------
        inp_tokens : torch.Tensor
            See BaseScorerInterface().
        memory : dict[str, scorer memory]
            The states of scorers for this timestep.
        attn : torch.Tensor
            See BaseScorerInterface().
        log_probs : torch.Tensor
            (batch_size x beam_size, vocab_size). The log probs at this timestep.
        beam_size : int
            The beam size.

        Returns
        ---------
        log_probs : torch.Tensor
            (batch_size x beam_size, vocab_size). Log probs updated by scorers.
        new_memory : dict[str, scorer memory]
            The updated states of scorers.
        """
        new_memory = dict()
        # score full candidates
        for k, impl in self.full_scorers.items():
            score, new_memory[k] = impl.score(inp_tokens, memory[k], None, attn)
            log_probs += score * self.weights[k]

        # select candidates from the results of full scorers for partial scorers
        _, candidates = log_probs.topk(
            int(beam_size * self.scorer_beam_scale), dim=-1
        )

        # score pruned tokens candidates
        for k, impl in self.partial_scorers.items():
            score, new_memory[k] = impl.score(
                inp_tokens, memory[k], candidates, attn
            )
            log_probs += score * self.weights[k]

        return log_probs, new_memory

    def permute_scorer_mem(self, memory, index, candidates):
        """Update memory variables of scorers to synchronize
        the memory index with the current output and perform
        batched beam search.

        Arguments
        ---------
        memory : dict[str, scorer memory]
            The states of scorers for this timestep.
        index : torch.Tensor
            (batch_size x beam_size). The index of the previous path.
        candidates : torch.Tensor
            (batch_size, beam_size). The index of the topk candidates.
        """
        for k, impl in self.full_scorers.items():
            # ctc scorer should always be scored by candidates
            if k == "ctc" or k == "kenlm":
                memory[k] = impl.permute_mem(memory[k], candidates)
                continue
            memory[k] = impl.permute_mem(memory[k], index)
        for k, impl in self.partial_scorers.items():
            memory[k] = impl.permute_mem(memory[k], candidates)
        return memory

    def reset_scorer_mem(self, x, enc_lens):
        """Reset memory variables for scorers.

        Arguments
        ---------
        x : torch.Tensor
            See BaseScorerInterface().
        wav_len : torch.Tensor
            See BaseScorerInterface().
        """
        memory = dict()
        for k, impl in {**self.full_scorers, **self.partial_scorers}.items():
            memory[k] = impl.reset_mem(x, enc_lens)
        return memory

    def _validate_scorer(self, scorer_names):
        """These error messages indicate scorers are not properly set.

        Arguments
        ---------
        scorer_names : list
            Prefix of scorers defined in speechbrain.decoders.scorer.
        """
        if len(self.weights) > len(scorer_names):
            raise ValueError(
                "The keys of weights should be named in {}".format(scorer_names)
            )

        if not 0.0 <= self.weights["ctc"] <= 1.0:
            raise ValueError("ctc_weight should not > 1.0 and < 0.0")

        if self.weights["ctc"] == 1.0:
            if "ctc" not in self.full_scorers.keys():
                raise ValueError(
                    "CTC scorer should be a full scorer when it's weight is 1.0"
                )
            if self.weights["coverage"] > 0.0:
                raise ValueError(
                    "Pure CTC scorer doesn't have attention weights for coverage scorer"
                )

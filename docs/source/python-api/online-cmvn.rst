Online CMVN
===========

Running (online) cepstral mean and variance normalization, a port of kaldi's
``OnlineCmvn`` / ``OnlineCmvnState``
(:kaldi:`feat/online-feature.h <src/feat/online-feature.h>`, see
:knf:`kaldi-native-fbank/csrc/online-cmvn.h`).

Unlike the offline CMVN of :doc:`cmvn` (which needs the whole utterance), this
class looks only at the **past**: each frame is normalized with statistics
accumulated from the frames seen so far.  This is what kaldi's online decoding
and i-vector front-ends use — if you train on features normalized this way you
must decode with the same kind of normalization (kaldi's "matched" online
training).

Offline vs online CMVN
----------------------

==============================  =================================================
offline (:doc:`cmvn`)           online (this page)
==============================  =================================================
stats over the whole            stats over the previous ``cmn_window`` frames
utterance (sees the future)     (streaming, past only)
output is zero-mean per         output only *approximately* centered,
utterance                       especially at the start
one shot                        one frame at a time
reversable (``reverse=True``)   frozen via :meth:`OnlineCmvn.freeze`
==============================  =================================================

In the steady state (middle of a long utterance) the class accumulates stats
from the previous ``cmn_window`` frames (default 600, i.e. 6 seconds) and uses
them to normalize the current frame.  At the **beginning of an utterance**,
when fewer than ``cmn_window`` frames of context exist, the stats are
augmented first with up to ``speaker_frames`` frames of speaker stats (from
previous utterances of the same speaker) and then with up to ``global_frames``
frames of global stats (from training data) — so the very first frames are
normalized against a "typical" distribution instead of almost nothing.

Usage
-----

.. code-block:: python

   import numpy as np
   import kaldi_native_fbank as knf

   dim = 40
   global_stats = ...          # [2, dim+1] float64, e.g. kaldi's global_cmvn.stats

   opts = knf.OnlineCmvnOptions()          # kaldi defaults
   state = knf.OnlineCmvnState(global_cmvn_stats=global_stats)
   cmvn = knf.OnlineCmvn(opts, state)

   # push raw feature frames one by one (or in batches)
   feats = np.random.randn(200, dim).astype(np.float32)
   out = cmvn.accept_features(feats)       # [200, dim] float32, normalized
   # out[t] = feats[t] - running_mean(t)   (mean normalization)

   # hand the speaker adaptation over to the next utterance:
   state = cmvn.get_state()
   cmvn2 = knf.OnlineCmvn(opts, state)
   out2 = cmvn2.accept_features(next_utterance)

Single frames use :meth:`OnlineCmvn.accept_feature`, which returns the
normalized frame of shape ``[dim]``.

Freezing the normalization
--------------------------

Calling :meth:`OnlineCmvn.freeze` fixes the normalization at the current
frame: later frames are normalized with the frozen stats and the running stats
stop updating.  This is kaldi's mechanism for estimating fMLLR (or anything
else) on top of a *fixed* CMVN.  The frozen state travels with
:class:`OnlineCmvnState`, so the next utterance of the speaker is normalized
the same way:

.. code-block:: python

   cmvn.accept_features(feats)
   cmvn.freeze()
   steady = cmvn.accept_features(more_feats)   # constant offset per dim

   state = cmvn.get_state()                    # carries frozen_state
   cmvn2 = knf.OnlineCmvn(opts, state)         # keeps the frozen normalization

Options
-------

:class:`OnlineCmvnOptions` mirrors kaldi's ``OnlineCmvnOptions`` (defaults
match):

======================  ========  ====================================================
option                  default   meaning
======================  ========  ====================================================
``cmn_window``          600       frames of sliding context for the running stats
``speaker_frames``      600       frames of previous utterance(s) of this speaker to
                                  back off to at utterance start (``<= cmn_window``)
``global_frames``       200       frames of global stats to back off to further
                                  (``<= speaker_frames``)
``normalize_mean``      True      subtract the running mean
``normalize_variance``  False     also divide by the running standard deviation
                                  (requires ``normalize_mean=True``)
``modulus``             20        kept for kaldi configuration compatibility; this
                                  port updates stats incrementally, so results are
                                  not affected
``ring_buffer_size``    20        kept for kaldi configuration compatibility (see
                                  ``modulus``)
``skip_dims``           ""        colon-separated dimensions to skip normalization
                                  of, e.g. ``"13:14:15"`` (useful for e.g. pitch
                                  features appended to the base features)
======================  ========  ====================================================

Formulas
--------

Stats have kaldi's 2 x (dim+1) layout (same as :doc:`cmvn`):
``row0 = [sum_1..sum_D, count]``, ``row1 = [sumsq_1..sumsq_D, 0]``.

For each accepted frame :math:`x_t` the running window stats are updated
(frame :math:`t` added, frame :math:`t - c` removed if it leaves the window,
where :math:`c` is the option ``cmn_window``):

.. math::

   \mathrm{count} = \sum_{s=t-c+1}^{t} 1, \qquad
   \mathrm{sum}_d = \sum_{s=t-c+1}^{t} x_{s,d}, \qquad
   \mathrm{sumsq}_d = \sum_{s=t-c+1}^{t} x_{s,d}^2

The window stats are then *smoothed*
with the speaker and global stats (this is what happens when
:math:`\mathrm{count} < c`, e.g. at the start of an utterance):

.. math::

   \mathrm{sum} \mathrel{+}= \frac{\min(c - n,\ s_f,\ n_s)}{n_s}\, \mathrm{sum}_{\mathrm{spk}},
   \quad\text{then}\quad
   \mathrm{sum} \mathrel{+}= \frac{\min(c - n',\ g_f)}{n_g}\, \mathrm{sum}_{\mathrm{glob}}

where :math:`n` is the window count, :math:`s_f`/``speaker_frames``,
:math:`g_f`/``global_frames``, :math:`n_s`/``n_g`` the counts of the speaker
and global stats (same for ``sumsq``).  Finally the frame is normalized with
the smoothed stats:

.. math::

   y_{t,d} = x_{t,d} - \frac{\mathrm{sum}_d}{\mathrm{count}}
   \qquad (\text{mean normalization})

   y_{t,d} = \frac{x_{t,d} - \mu_d}{\sqrt{\sigma_d^2}}, \quad
   \mu_d = \frac{\mathrm{sum}_d}{\mathrm{count}}, \quad
   \sigma_d^2 = \frac{\mathrm{sumsq}_d}{\mathrm{count}} - \mu_d^2
   \qquad (\text{+ variance normalization})

Note the frame being normalized **is included** in the stats that normalize it
(like kaldi).  Variance is floored at :math:`10^{-20}` before the square root.

Correspondence with kaldi
-------------------------

=====================================================  =============================================================
kaldi (``feat/online-feature.h``)                      kaldi-native-fbank
=====================================================  =============================================================
``OnlineCmvnOptions``                                  :class:`OnlineCmvnOptions`
``OnlineCmvnState``                                    :class:`OnlineCmvnState`
``OnlineCmvn`` (pull-style filter: ``GetFrame(t)``)    :class:`OnlineCmvn` (push-style: ``accept_*``)
``OnlineCmvn::Freeze(t)``                              :meth:`OnlineCmvn.freeze`
``OnlineCmvn::GetState(t, &s)``                        :meth:`OnlineCmvn.get_state` (speaker stats are accumulated
                                                       incrementally; same result)
``OnlineCmvn::SetState(s)``                            pass the state to the :class:`OnlineCmvn` constructor
``SmoothOnlineCmvnStats``                              :meth:`OnlineCmvn.smooth_online_cmvn_stats`
=====================================================  =============================================================

Parity with kaldi is tested frame-by-frame against
:kaldi:`apply-cmvn-online <src/online2bin/apply-cmvn-online.cc>` in
``tests/parity/parity_online_cmvn.py`` (several option sets including
speaker-state handover agree to about 1e-5).

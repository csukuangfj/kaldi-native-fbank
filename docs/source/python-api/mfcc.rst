MFCC
====

Mel-frequency cepstral coefficients, kaldi-compatible
(:knf:`kaldi-native-fbank/csrc/feature-mfcc.h`).  Numerically equal to kaldi's
:kaldi:`compute-mfcc-feats <src/featbin/compute-mfcc-feats.cc>`
for identical parameters.

Usage
-----

.. code-block:: python

   import numpy as np
   import kaldi_native_fbank as knf

   samples = ...  # 1-D float32 waveform, raw int16 amplitude scale
   sample_rate = 16000

   opts = knf.MfccOptions()
   opts.frame_opts.samp_freq = sample_rate
   opts.frame_opts.snip_edges = True
   opts.frame_opts.dither = 0.0          # use 0 for deterministic output
   opts.mel_opts.num_bins = 23
   opts.mel_opts.low_freq = 20
   opts.mel_opts.high_freq = 7800
   opts.num_ceps = 13
   opts.use_energy = False
   opts.cepstral_lifter = 22

   mfcc = knf.OnlineMfcc(opts)
   mfcc.accept_waveform(sample_rate, samples)
   mfcc.input_finished()
   num_frames = mfcc.num_frames_ready
   frame = mfcc.get_frame(0)             # list of floats, dim = num_ceps

**Important**: feed samples at **raw int16 amplitude** (do NOT divide by
32768).  kaldi's wav reader keeps raw magnitudes; normalizing scales the
energy by 32768^2 and shifts C0.

Equivalent in kaldi
-------------------

.. code-block:: bash

   # kaldi (dither off for deterministic output)
   compute-mfcc-feats --dither=0.0 --use-energy=false --num-ceps=13 \\
     --num-mel-bins=23 --cepstral-lifter=22 --low-freq=20 --high-freq=7800 \\
     --sample-frequency=16000 --snip-edges=true scp:wav.scp ark:mfcc.ark

Mapping
^^^^^^^

==============================  =====================================
kaldi option                    python API                           
==============================  =====================================
``--num-ceps``                  ``MfccOptions.num_ceps``             
``--num-mel-bins``              ``MfccOptions.mel_opts.num_bins``    
``--cepstral-lifter``           ``MfccOptions.cepstral_lifter``      
``--use-energy``                ``MfccOptions.use_energy``           
``--energy-floor``              ``MfccOptions.energy_floor``         
``--sample-frequency``          ``MfccOptions.frame_opts.samp_freq`` 
``--snip-edges``                ``MfccOptions.frame_opts.snip_edges``
``--dither``                    ``MfccOptions.frame_opts.dither``    
``--low-freq``/``--high-freq``  ``mel_opts.low_freq``/``high_freq``  
``--preemphasis-coefficient``   ``frame_opts.preemph_coeff``         
==============================  =====================================


With ``dither=0`` and identical parameters the output matches kaldi
**value-by-value** (verified against :kaldi:`compute-mfcc-feats <src/featbin/compute-mfcc-feats.cc>` output; residual
diff ~1e-4 at float rounding level).

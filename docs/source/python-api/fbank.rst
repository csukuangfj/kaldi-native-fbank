Fbank
=====

Filter-bank features, kaldi-compatible
(:knf:`kaldi-native-fbank/csrc/feature-fbank.h`).  Numerically equal to kaldi's
:kaldi:`compute-fbank-feats <src/featbin/compute-fbank-feats.cc>`
for identical parameters.

Usage
-----

.. code-block:: python

   import kaldi_native_fbank as knf

   opts = knf.FbankOptions()
   opts.frame_opts.samp_freq = 16000
   opts.frame_opts.snip_edges = True
   opts.frame_opts.dither = 0.0
   opts.mel_opts.num_bins = 23
   opts.mel_opts.low_freq = 20
   opts.mel_opts.high_freq = 7800
   opts.use_energy = False
   opts.use_log_fbank = True
   opts.use_power = True

   fbank = knf.OnlineFbank(opts)
   fbank.accept_waveform(16000, samples)
   fbank.input_finished()
   frame = fbank.get_frame(0)   # dim = num_bins

Equivalent in kaldi
-------------------

.. code-block:: bash

   # kaldi
   compute-fbank-feats --dither=0.0 --num-mel-bins=23 --low-freq=20 \\
     --high-freq=7800 --sample-frequency=16000 --snip-edges=true \\
     scp:wav.scp ark:fbank.ark

Mapping
^^^^^^^

==============================  ======================================
kaldi option                    python API                            
==============================  ======================================
``--num-mel-bins``              ``FbankOptions.mel_opts.num_bins``    
``--use-energy``                ``FbankOptions.use_energy``           
``--use-log-fbank``             ``FbankOptions.use_log_fbank``        
``--use-power``                 ``FbankOptions.use_power``            
``--sample-frequency``          ``FbankOptions.frame_opts.samp_freq`` 
``--snip-edges``                ``FbankOptions.frame_opts.snip_edges``
``--dither``                    ``FbankOptions.frame_opts.dither``    
``--low-freq``/``--high-freq``  ``mel_opts.low_freq``/``high_freq``   
==============================  ======================================


# Results

Speech vs non-speech accuracy. tested on only 300 clips (150 speech, 150
nonspeech, half CommonVoice half VIVAE) for each run.
But tested across all 1000+ vIVAE files, 300 commonvoice samples as well plus noisy audios

## Clean audio(manifests_raw)

- AST: 99% accuracy. Out of 300 clips, only 4 wrong (3 nonspeech
  called speech, 1 speech called nonspeech)
- PANNs: 97% accuracy. 8 wrong (6 nonspeech called speech, 2 speech
  called nonspeech)

Both models nearly perfect here. Almost no mistakes.

## Augmented audio (manifests_aug)

- AST: 96% accuracy. 11 wrong (2 nonspeech called speech, 9 speech
  called nonspeech)
- PANNs: 96% accuracy. 12 wrong (3 nonspeech called speech, 9 speech
  called nonspeech)

Both models drop a bit here and make the same kind of mistakes: 9 real
speech clips(all CommonVoice) get called "nonspeech." I checked
it's the exact same 9 clips for both models. So its not an AST or
PANNs problem, it's the VAD struggling on these
specific augmented clips. 



## notes
AST is a bit stronger than PANNs on clean audio. Both are about equal
on augmented audio. Both are good with 95% accuracy either way. 
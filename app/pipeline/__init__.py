"""Pipeline di traduzione: sorgente (ASR) -> traduzione -> TTS.

Le astrazioni in ``base`` hanno due implementazioni:
  - ``mock``  : nessun modello, per provare il sistema end-to-end
  - reale     : faster-whisper (ASR), NLLB (MT), Piper (TTS)

La selezione avviene in ``factory`` in base a ``engine.mode`` della config.
"""

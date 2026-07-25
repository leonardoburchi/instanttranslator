"""Instant Translator — traduzione audio multicanale in tempo reale.

Backend di regia (FastAPI) + frontend web. Cattura flussi audio
multicanale, li separa e per ogni canale esegue ASR -> traduzione -> TTS,
distribuendo sottotitoli e audio agli ascoltatori nella lingua scelta.
"""

__version__ = "0.1.0"

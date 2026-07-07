from .common import *
from .utils import *
from .main_page import create_main_page
from .long_text_page import create_long_text_page

__all__ = [
    'create_main_page',
    'create_long_text_page',
    'i18n',
    'tts',
    'LANGUAGES',
    'EMO_CHOICES_ALL',
    'EMO_CHOICES_OFFICIAL',
    'MODE',
    'cmd_args',
]
# Facade DB Module para mantener compatibilidad
#
# [P3-NEW-STAR-IMPORTS-AUDIT · 2026-05-15] Los 6 `from db_X import *` colapsan
# el namespace; si dos módulos definen un símbolo top-level con el mismo
# nombre, el último import gana silenciosamente. Refactor a imports
# explícitos es invasivo (100+ exports). En su lugar, el test parser-based
# `tests/test_p3_new_star_imports_audit.py` parsea con AST y falla si
# detecta conflict — defense-in-depth sin tocar prod.
import os
import logging

from db_core import *
from db_profiles import *
from db_chat import *
from db_plans import *
from db_facts import *
from db_inventory import *

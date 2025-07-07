import pathlib

from attrdict import AttrDict
from environs import Env

env = Env()
env.read_env('.env')
ENV_FILE = env.path('ENV_FILE', default=None)
if ENV_FILE:
    ENV_FILE = ENV_FILE.expanduser().resolve()
    env.read_env(str(ENV_FILE))

# LLM Provider Selection
LLM = AttrDict()
LLM.PROVIDER = env.str('LLM_PROVIDER', default=None)  # 'openai', 'anthropic', 'local', or None for auto-select
LLM.DEFAULT_MAX_FILE_SIZE = env.int('LLM_DEFAULT_MAX_FILE_SIZE', default=250000)

# OpenAI Configuration
OPENAI = AttrDict()
OPENAI.ORGANIZATION = env.str('OPENAI_ORGANIZATION', default=None)
OPENAI.API_KEY = env.str('OPENAI_API_KEY', default=None)
OPENAI.MODEL = env.str('OPENAI_MODEL', default='gpt-4')
OPENAI.TEMPERATURE = env.float('OPENAI_TEMPERATURE', default=0.3)
OPENAI.MAX_RETRIES = env.int('OPENAI_MAX_RETRIES', default=3)
OPENAI.TIMEOUT = env.int('OPENAI_TIMEOUT', default=90)

# Anthropic Configuration
ANTHROPIC = AttrDict()
ANTHROPIC.API_KEY = env.str('ANTHROPIC_API_KEY', default=None)
ANTHROPIC.MODEL = env.str('ANTHROPIC_MODEL', default='claude-3-sonnet-20240229')
ANTHROPIC.BASE_URL = env.str('ANTHROPIC_BASE_URL', default='https://api.anthropic.com/v1/messages')
ANTHROPIC.TIMEOUT = env.int('ANTHROPIC_TIMEOUT', default=90)

# Local LLM Configuration (Ollama, LM Studio, etc.)
LOCAL_LLM = AttrDict()
LOCAL_LLM.BASE_URL = env.str('LOCAL_LLM_BASE_URL', default='http://localhost:11434')
LOCAL_LLM.MODEL = env.str('LOCAL_LLM_MODEL', default='codellama')
LOCAL_LLM.TIMEOUT = env.int('LOCAL_LLM_TIMEOUT', default=120)

# Git Configuration
GIT = AttrDict()
DEFAULT_GIT_PATH = pathlib.Path('.').resolve()
GIT.DEFAULT_REPO_PATH = env.path('GIT_DEFAULT_REPO_PATH', default=DEFAULT_GIT_PATH)

# Output Configuration
OUTPUT = AttrDict()
OUTPUT.DEFAULT_DIR = env.str('OUTPUT_DEFAULT_DIR', default='repo_intel_output')
OUTPUT.VERBOSE = env.bool('OUTPUT_VERBOSE', default=False)

# Markdown Bundle Configuration
MARKDOWN = AttrDict()
MARKDOWN.DEFAULT_OUTPUT = env.str('MARKDOWN_DEFAULT_OUTPUT', default='bundle.md')
MARKDOWN.EXCLUDE_PATTERNS = env.list('MARKDOWN_EXCLUDE_PATTERNS', default=[
    '.egg-info', '.aws-sam', '.git', '__pycache__', '__tests__',
    'node_modules', 'venv', '.env', '.idea', '.yarn', '.vscode'
])

# AWS Configuration for Glue
AWS = AttrDict()
AWS.REGION = env.str('AWS_REGION', default=None)
AWS.PROFILE = env.str('AWS_PROFILE', default=None)

# Glue Bundle Configuration
GLUE = AttrDict()
GLUE.DEFAULT_OUTPUT = env.str('GLUE_DEFAULT_OUTPUT', default='glue_documentation.md')
GLUE.EXCLUDE_DATABASES = env.list('GLUE_EXCLUDE_DATABASES', default=[])
GLUE.EXCLUDE_TABLES = env.list('GLUE_EXCLUDE_TABLES', default=[])

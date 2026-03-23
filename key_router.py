import os
import logging
from dotenv import load_dotenv

load_dotenv()

from google import genai
from google.genai.errors import APIError

logger = logging.getLogger(__name__)

class KeyRouter:
    def __init__(self, name, env_vars, default_env_vars=None):
        self.name = name
        def extract_keys(var_names):
            extracted = []
            for v in var_names:
                val = os.getenv(v)
                if val:
                    extracted.extend([k.strip() for k in val.replace('\n', ',').split(',') if k.strip()])
                
                prefix = v + "_"
                matching_keys = [k for k in os.environ.keys() if k.startswith(prefix)]
                matching_keys.sort()
                for mk in matching_keys:
                    mval = os.environ[mk]
                    if mval:
                        extracted.extend([k.strip() for k in mval.replace('\n', ',').split(',') if k.strip()])
            
            seen = set()
            return [x for x in extracted if not (x in seen or seen.add(x))]

        self.keys = extract_keys(env_vars)
        if not self.keys and default_env_vars:
            self.keys = extract_keys(default_env_vars)
            
        self.idx = 0

    def get_key(self):
        if not self.keys:
            return None
        return self.keys[self.idx]

    def rotate(self):
        if self.keys:
            self.idx = (self.idx + 1) % len(self.keys)
            logger.error(f"[{self.name} Router] Rotated API key to index {self.idx}")

    def reset(self):
        self.idx = 0
        logger.error(f"[{self.name} Router] 游标已强制重置为 0")

    def get_client(self, **kwargs):
        key = self.get_key()
        if not key:
            raise ValueError(f"No API keys configured for {self}.")
        return genai.Client(api_key=key, **kwargs)

    def get_client_by_index(self, idx, **kwargs):
        """用指定 index 的 key 创建 client，不影响当前游标"""
        if not self.keys:
            raise ValueError(f"No API keys configured for {self.name}.")
        if idx < 0 or idx >= len(self.keys):
            raise ValueError(f"[{self.name}] key index {idx} 超出范围 (共 {len(self.keys)} 个 key, 0~{len(self.keys)-1})")
        key = self.keys[idx]
        return genai.Client(api_key=key, **kwargs)

    def execute_with_fixed_key(self, key_index, action_fn, **get_client_kwargs):
        """用指定 index 的 key 执行单次调用，不影响当前游标，不做轮转"""
        client = self.get_client_by_index(key_index, **get_client_kwargs)
        key_str = self.keys[key_index]
        masked_key = key_str[:12] + "..." + key_str[-4:] if len(key_str) > 16 else key_str
        logger.error(f"[{self.name} Router] 手动指定 Key (index {key_index}): {masked_key}")
        return action_fn(client)

    def execute_with_retry(self, action_fn, get_client_kwargs=None, on_rotate=None):
        """
        Executes action_fn(client) and wraps it with rotation logic.
        action_fn should be a callable that takes the client as its only argument.
        If a quota/auth error occurs, it rotates and retries up to len(keys) times.
        """
        if get_client_kwargs is None:
            get_client_kwargs = {}
            
        attempts = 0
        max_attempts = len(self.keys) if self.keys else 1
        
        while attempts < max_attempts:
            client = self.get_client(**get_client_kwargs)
            key_str = self.get_key() or "unknown"
            masked_key = key_str[:12] + "..." + key_str[-4:] if len(key_str) > 16 else key_str
            logger.error(f"[{self.name} Router] 正在调用 Gemini API，当前使用的 Key (游标 {self.idx}): {masked_key}")
            
            try:
                res = action_fn(client)
                logger.error(f"[{self.name} Router] 调用成功 (游标 {self.idx}): {masked_key}")
                return res
            except Exception as e:
                logger.error(f"[{self.name} Router] 调用报错 (游标 {self.idx}): {masked_key} - 报错信息: {e}")
                # APIError in google.genai has .code
                is_quota_or_auth = False
                if isinstance(e, APIError):
                    if e.code in (400, 403, 429, 500, 502, 503):
                        is_quota_or_auth = True
                    # Also fallback on 'Resource has been exhausted (e.g. check quota)'
                    if 'quota' in str(e).lower() or 'exhausted' in str(e).lower() or 'invalid' in str(e).lower():
                        is_quota_or_auth = True
                elif '429' in str(e) or '403' in str(e) or 'exhausted' in str(e).lower():
                    is_quota_or_auth = True
                    
                if is_quota_or_auth and attempts < max_attempts - 1:
                    logger.error(f"API Error ({e}), rotating key...")
                    self.rotate()
                    if on_rotate:
                        on_rotate(self.get_client(**get_client_kwargs))
                    attempts += 1
                else:
                    raise e
        raise Exception("All API keys failed.")

chat_router = KeyRouter('Chat', ['GOOGLE_API_KEY', 'GEMINI_API_KEY'])
embed_router = KeyRouter('Embed', ['GEMINI_API_KEY_EMBED'], default_env_vars=['GOOGLE_API_KEY', 'GEMINI_API_KEY'])
image_router = KeyRouter('Image', ['GEMINI_API_KEY_IMAGE'], default_env_vars=['GOOGLE_API_KEY', 'GEMINI_API_KEY'])

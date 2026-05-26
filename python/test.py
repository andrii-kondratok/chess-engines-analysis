import os
from dotenv import load_dotenv
load_dotenv()

token = os.getenv("HF_TOKEN")
if token:
    print(f"✓ Token loaded: {token[:10]}...{token[-4:]}")
else:
    print("✗ Token not found!")
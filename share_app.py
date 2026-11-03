"""Free Gradio share link; inference stays on the machine running this file."""
import json
import os
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
runtime = ROOT / '.runtime'
runtime.mkdir(exist_ok=True)
os.environ.setdefault('HF_HOME', str(runtime / 'huggingface'))
os.environ.setdefault('GRADIO_TEMP_DIR', str(runtime / 'uploads'))
os.environ.setdefault('GRADIO_ANALYTICS_ENABLED', 'False')
os.environ.setdefault('NO_PROXY', '127.0.0.1,localhost,::1')
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')

try:
    import truststore
except ImportError:
    pass
else:
    truststore.inject_into_ssl()

from app import app

if __name__ == '__main__':
    port = int(os.environ.get('GRADING_SHARE_PORT', '7862'))
    app.queue(default_concurrency_limit=1, max_size=20).launch(
        server_name='127.0.0.1', server_port=port,
        share=True, show_api=False, max_file_size='20mb',
        prevent_thread_lock=True)
    if not app.share_url:
        app.close()
        raise RuntimeError('未能创建 Gradio 公网分享链接，请检查网络和运行日志。')
    record = {
        'status': 'running', 'mode': 'local_inference_gradio_share',
        'url': app.share_url, 'local_url': f'http://127.0.0.1:{port}',
        'pid': os.getpid(), 'started_at': datetime.now(timezone.utc).isoformat(),
        'requires_local_computer_online': True,
        'temporary_link': True, 'cloud_hosted': False,
        'models': 'independently_trained_prompt3_prompt4_with_pretrained_trocr',
    }
    status = Path(os.environ.get('GRADING_SHARE_STATUS', str(runtime / 'gradio-share.json')))
    status.parent.mkdir(parents=True, exist_ok=True)
    status.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
    print('GRADING_PUBLIC_SHARE ' + json.dumps(record, ensure_ascii=False), flush=True)
    app.block_thread()

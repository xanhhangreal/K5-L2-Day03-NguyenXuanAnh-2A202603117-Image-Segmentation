"""Local Docker-only bridge to CVAT's real REST views; no persistent credentials."""
import base64
import json
import os
import sys
from pathlib import Path
from urllib.parse import urlsplit

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'cvat.settings.production')
sys.path.insert(0, '/home/django')
import django
django.setup()
from django.test import override_settings
from django.core.files.uploadedfile import SimpleUploadedFile
from rest_framework.test import APIClient
from cvat.apps.engine.models import Task


def main():
    arg = json.loads(Path(sys.argv[1]).read_text())
    owner = Task.objects.get(pk=3).owner
    if owner.username != 'anhnguyenkaz':
        raise RuntimeError('Unexpected Easy task owner')
    client = APIClient()
    client.force_authenticate(user=owner)
    path = arg['path']
    parts = urlsplit(path)
    path = parts.path + ('?' + parts.query if parts.query else '')
    if not path.startswith('/api/'):
        raise ValueError('Only local CVAT API paths are permitted')
    method = arg.get('method', 'GET').lower()
    data = arg.get('data', {})
    fmt = 'json'
    if arg.get('files'):
        fmt = 'multipart'
        for i, file in enumerate(arg['files']):
            p = Path(file)
            if not p.is_relative_to('/tmp/lab3-auto'):
                raise ValueError('Upload outside staging directory')
            data[f'client_files[{i}]'] = SimpleUploadedFile(p.name, p.read_bytes(), content_type='image/jpeg')
    # Only this short-lived CLI process uses a streaming download backend.
    with override_settings(SENDFILE_BACKEND='django_sendfile.backends.development'):
        response = getattr(client, method)(path, data=data, format=fmt, HTTP_HOST='localhost')
        if arg.get('download') and response.status_code == 200:
            out = Path(f'/tmp/lab3-auto-download-{os.getuid()}.zip')
            with out.open('wb') as stream:
                if response.streaming:
                    for chunk in response.streaming_content:
                        stream.write(chunk)
                else:
                    stream.write(response.content)
            payload = {'file': str(out), 'size': out.stat().st_size}
        else:
            try:
                payload = response.json()
            except Exception:
                payload = response.content.decode(errors='replace')[:4000]
    print('LAB3_RESULT=' + json.dumps({'status': response.status_code, 'data': payload}, ensure_ascii=False))


if __name__ == '__main__':
    main()

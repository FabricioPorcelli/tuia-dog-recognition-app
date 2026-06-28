import requests
from pathlib import Path

external_dir = Path('data/test_external')
external_dir.mkdir(exist_ok=True)

# Descargar imagenes individuales desde dog.ceo (API publica de razas)
dog_urls = {
    'dog1.jpg':           'https://placedog.net/640/640?random',
    'dog2.jpg':           'https://placedog.net/640/640?random',
    'dog3.jpg':            'https://placedog.net/640/640?random',
    'dog4.jpg':            'https://placedog.net/640/640?random',
    'dog5.jpg':            'https://placedog.net/640/640?random',
    'dog6.jpg':            'https://placedog.net/640/640?random',
    'dog7.jpg':            'https://placedog.net/640/640?random',
    'dog8.jpg':            'https://placedog.net/640/640?random',
    'dog9.jpg':            'https://placedog.net/640/640?random',
    'dog10.jpg':            'https://placedog.net/640/640?random',
    'dog11.jpg':           'https://placedog.net/640/640?random',
    'dog12.jpg':           'https://placedog.net/640/640?random',
    'dog13.jpg':            'https://placedog.net/640/640?random',
    'dog14.jpg':            'https://placedog.net/640/640?random',
    'dog15.jpg':            'https://placedog.net/640/640?random',
    'dog16.jpg':            'https://placedog.net/640/640?random',
    'dog17.jpg':            'https://placedog.net/640/640?random',
    'dog18.jpg':            'https://placedog.net/640/640?random',
    'dog19.jpg':            'https://placedog.net/640/640?random',
    'dog20.jpg':            'https://placedog.net/640/640?random',
}

for fname, url in dog_urls.items():
    dest = external_dir / fname
    if not dest.exists():
        try:
            resp = requests.get(url, timeout=15)
            resp.raise_for_status()
            dest.write_bytes(resp.content)
            print(f'Descargado: {fname}')
        except Exception as e:
            print(f'Error: {url} - {e}')
    else:
        print(f'Ya existe: {fname}')
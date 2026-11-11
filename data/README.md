# Данные

`data/raw/` не коммитится. Как восстановить:

```bash
mkdir -p data/raw && cd data/raw
curl -sSL https://codeload.github.com/pszemraj/megalodon-enwik8/tar.gz/HEAD | tar xz --wildcards '*/data/enwik8.gz' --strip-components=2
gunzip -k enwik8.gz && sha256sum enwik8   # должно совпасть с data/enwik8.sha256
```

Сплит (стандартный для enwik8): train = байты [0, 90M), valid = [90M, 95M), test = [95M, 100M).
Скрипты читают `data/raw/enwik8` напрямую по смещениям. Test открывается один раз, на зачётном прогоне (`docs/02` п.1).

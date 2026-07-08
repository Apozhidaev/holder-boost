# holder-boost

**holder-boost** — сервис предсказания исхода бинарного маркета Polymarket (**Yes** / **No**)
с помощью CatBoost по поведению холдеров.

Модель обучается на resolved-маркетах: для каждого дня до экспирации (`days_left`) и набора
`proxy_wallet` строится вектор признаков из распределения позиций холдеров по исходам.
Целевая переменная — **`resolved_outcome_index`** (0 = Yes, 1 = No).

**Что делает проект:**

1. **`sync`** — загружает resolved-маркеты и holder-distribution из poly-house в Mongo.
2. **`train`** — обучает CatBoost на исторических данных и сохраняет артефакты в `.models/`.
3. **`serve`** — REST API **`POST /predict`**: на вход `days_left` и текущие позиции холдеров,
   на выход `predict_probs` — `[P(Yes), P(No)]`.

## Принцип работы

Каталог resolved-маркетов живёт в Mongo (**`markets`**): документ по Gamma **`id`**
(`ResolvedMarket`: `condition_id`, `resolved_outcome_index`, `holders`, `max_days_left`, `tags` и т.д.).
Наполняется командой **`sync`** (см. раздел **`sync`** ниже; при необходимости запускайте повторно).

## Требования

- Python 3.12+
- [`uv`](https://github.com/astral-sh/uv)
- Доступный экземпляр MongoDB

## Установка

```bash
uv sync
```

Создаётся виртуальное окружение проекта и ставятся runtime- и dev-зависимости
из `pyproject.toml` (включая CatBoost, FastAPI и uvicorn).

## Конфигурация

Настройки лежат в `config.toml` в корне репозитория.

Секции:

- `[mongo]` — `uri`, `db`
- `[http]` — `timeout_seconds`
- `[logging]` — `level`, `format` (`json` / `pretty`)
- `[poly_house]` — `base_url` (по умолчанию `http://127.0.0.1:8000`); **`sync`**
- `[sync]` — загрузка resolved-маркетов в **`markets`** из poly-house:
  - `tag_ids` — числовые tag id (`list[int]`); JSON body на **`GET /markets`**; пустой список — без фильтра по тегам
  - `neg_risk` — query **`neg_risk`** на **`GET /markets`**; по умолчанию **`false`**
- `[train]` — обучение CatBoost на holder-distribution из **`markets`** (предсказание **`resolved_outcome_index`**, 0 = Yes, 1 = No):
  - `model_dir` — каталог артефактов (по умолчанию `.models/`)
  - `neg_risk` — фильтр маркетов при обучении
  - `min_wallet_count` — минимум вхождений `proxy_wallet` в маркетах для колонки `proxy_wallet_*`
  - `n_trials`, `n_folds`, `random_state` — Optuna и CV
- `[api]` — REST API (**`serve`**):
  - `host`, `port` — адрес сервера (по умолчанию `127.0.0.1:8003`); модель из **`[train].model_dir`**

## Команды

### `sync`

Один вызов без аргументов:

1. Загрузка closed-маркетов через poly-house **`GET /markets`** (offset-пагинация, `limit` до 500);
   query **`neg_risk`** из **`[sync]`**; опционально body **`tag_ids`** из **`[sync]`**; теги — **`GET /tags/{id}`** для id из маркетов;
   парсинг в **`ResolvedMarket`**.
2. Для **новых** маркетов (нет документа по **`id`**) — **`GET /activity/holder-distribution`**: распределение холдеров сохраняется в **`holders`** — `[{proxy_wallet, distribution: {outcome_index: count}}]`; **`max_days_left`** — горизонт в днях из того же ответа.
   Требуется запущенный сервис на **`poly_house.base_url`** (по умолчанию `http://127.0.0.1:8000`).
3. **Insert** в коллекцию **`markets`** по **`id`** (уже существующие документы не перезаписываются;
   повторный **`sync`** не backfill'ит holders).

При ошибке HTTP poly-house синхронизация завершается кодом **1** (stderr и лог). Битые строки
парсинга пропускаются с warning, не валят весь прогон. Требуется запущенный poly-house на **`poly_house.base_url`**.

```bash
uv run holder-boost sync
uv run holder-boost sync --max-pages 2  # ограничить пагинацию /markets (для тестов)
```

### `train`

Один вызов (только Mongo, без HTTP):

1. Загрузка всех маркетов из **`markets`** с **`neg_risk`** из **`[train]`**
2. Разворачивание строк `(market, day)` для `day = 0..max_days_left`; фичи: **`days_left`** и **`proxy_wallet_{address}`** (`distribution[day]`, иначе `-1`); строки, где все wallet-колонки `-1`, пропускаются
3. Label: **`resolved_outcome_index`**
4. Optuna + 5-fold CV (или **`--skip-tuning`** для быстрого прогона)
5. Сохранение в **`model_dir`**: `catboost.cbm`, `wallets.pkl`, `metadata.json`, `feature_importance.csv`

Требует предварительного **`sync`**.

```bash
uv run holder-boost train
uv run holder-boost train --skip-tuning   # без Optuna
```

### `serve`

FastAPI-сервер с **`POST /predict`** (требует обученную модель в **`[train].model_dir`**).

Request:
```json
{
  "days_left": 3,
  "holders": [
    {"proxy_wallet": "0xabc...", "outcome_index": 1},
    {"proxy_wallet": "0xdef...", "outcome_index": 0}
  ]
}
```

Response:
```json
{
  "predict_probs": [0.31, 0.69]
}
```

`predict_probs` — `[P(resolved_outcome_index=0), P(resolved_outcome_index=1)]` (`[P_yes, P_no]`). Кошельки из модели, не переданные в `holders`, получают `-1` в фичах.

```bash
uv run holder-boost serve
curl http://127.0.0.1:8003/health
curl -X POST http://127.0.0.1:8003/predict \
  -H 'Content-Type: application/json' \
  -d '{"days_left": 1, "holders": [{"proxy_wallet": "0x...", "outcome_index": 1}]}'
```

OpenAPI: `http://127.0.0.1:8003/docs`

## Линтинг

```bash
uv run ruff check .
uv run ruff format .
```

# holder-boost

**holder-boost** — сервис предсказания исхода бинарного маркета Polymarket (**Yes** / **No**)
с помощью CatBoost по поведению холдеров.

Модель обучается на resolved-маркетах: для каждого дня до экспирации (`days_left`) и набора
`proxy_wallet` строится вектор признаков из распределения позиций холдеров по исходам.
Целевая переменная — **`resolved_outcome_index`** (0 = Yes, 1 = No).

**Что делает проект:**

1. **`dataset`** — загружает resolved-маркеты и holder-distribution из poly-house по **`[train].tag_id`**
   и сохраняет training frame в **`.datasets/{tag_id}/`** (parquet).
2. **`train`** — обучает CatBoost на сохранённом dataset и пишет артефакты в **`.models/{tag_id}/`**.
3. **`serve`** — REST API **`POST /predict`**: на вход `tag_id`, `days_left` и текущие позиции холдеров,
   на выход `predict_probs` — `[P(Yes), P(No)]`.

## Принцип работы

Pipeline в два шага:

1. **`dataset`** — HTTP к poly-house, сборка строк `(market, day)`, сохранение в **`.datasets/{tag_id}/`**
2. **`train`** — load parquet, CatBoost, артефакты в **`.models/{tag_id}/`**

Повторные прогоны **`train`** (Optuna, `--skip-tuning`) не ходят в poly-house.

## Требования

- Python 3.12+
- [`uv`](https://github.com/astral-sh/uv)
- Запущенный poly-house на **`poly_house.base_url`** (для **`dataset`**)

## Установка

```bash
uv sync
```

Создаётся виртуальное окружение проекта и ставятся runtime- и dev-зависимости
из `pyproject.toml` (включая CatBoost, FastAPI и uvicorn).

## Конфигурация

Настройки лежат в `config.toml` в корне репозитория.

Секции:

- `[http]` — `timeout_seconds`
- `[logging]` — `level`, `format` (`json` / `pretty`)
- `[poly_house]` — `base_url` (по умолчанию `http://127.0.0.1:8000`); **`dataset`**
- `[train]` — обучение CatBoost (предсказание **`resolved_outcome_index`**, 0 = Yes, 1 = No):
  - `tag_id` — числовой tag id; JSON body на **`GET /markets`**
  - `model_dir` — базовый каталог артефактов (по умолчанию `.models/`); модель тега — **`{model_dir}/{tag_id}/`**
  - `dataset_dir` — базовый каталог dataset (по умолчанию `.datasets/`); parquet тега — **`{dataset_dir}/{tag_id}/`**
  - `neg_risk` — query **`neg_risk`** на **`GET /markets`**
  - `exclude_tag_ids` — опциональный query **`exclude_tag_ids`** на **`GET /markets`** (список tag id для исключения; по умолчанию пустой)
  - `min_wallet_count` — минимум вхождений `proxy_wallet` в маркетах для колонки с адресом кошелька
  - `n_trials`, `n_folds`, `random_state` — Optuna и CV
- `[api]` — REST API (**`serve`**):
  - `host`, `port` — адрес сервера (по умолчанию `127.0.0.1:8080`); модели из **`[train].model_dir/{tag_id}/`**

## Команды

### `dataset`

Один вызов без аргументов (тег из **`[train].tag_id`**):

1. Загрузка closed-маркетов через poly-house **`GET /markets`** (offset-пагинация, `limit` до 500);
   body = `[tag_id]` из **`[train]`**; query **`neg_risk`** и опционально **`exclude_tag_ids`** из **`[train]`**
2. Для каждого маркета — **`GET /activity/holders`**, затем **`GET /activity/holder-distribution`**
   на каждый `proxy_wallet`; `max_days_left` маркета = max `max_days_left` холдеров (где `>= 0`), иначе `-1` (неизвестно)
3. Разворачивание строк `(market, day)` для `day = 0..max_days_left` (маркеты с `max_days_left < 0` пропускаются); фичи: **`days_left`** и адрес кошелька (`distribution[day]`, иначе `-1`); строки, где все wallet-колонки `-1`, пропускаются
4. Сохранение в **`{dataset_dir}/{tag_id}/`**: `train.parquet`, `metadata.json`

При ошибке HTTP poly-house сборка dataset завершается кодом **1**. Ошибка holders/distribution
для отдельного маркета — warning и skip; ошибка одного кошелька — warning и skip кошелька.

```bash
uv run holder-boost dataset
uv run holder-boost dataset --max-pages 2  # ограничить пагинацию /markets (для тестов)
```

### `train`

Один вызов без аргументов (требует готовый **`{dataset_dir}/{tag_id}/`**):

1. Load `train.parquet` + проверка `metadata.json` (`tag_id`, `min_wallet_count`)
2. Optuna + 5-fold CV (или **`--skip-tuning`** для быстрого прогона)
3. Сохранение в **`{model_dir}/{tag_id}/`**: `catboost.cbm`, `wallets.pkl`, `metadata.json`, `feature_importance.csv`

```bash
uv run holder-boost train
uv run holder-boost train --skip-tuning
```

### `serve`

FastAPI-сервер с **`GET /tags`**, **`GET /features/{tag_id}`** и **`POST /predict`** (модели в **`.models/{tag_id}/`**).

Request:
```json
{
  "tag_id": 102486,
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
curl http://127.0.0.1:8003/tags
curl http://127.0.0.1:8003/features/102486
curl -X POST http://127.0.0.1:8003/predict \
  -H 'Content-Type: application/json' \
  -d '{"tag_id": 102486, "days_left": 1, "holders": [{"proxy_wallet": "0x...", "outcome_index": 1}]}'
```

OpenAPI: `http://127.0.0.1:8003/docs`

## Линтинг

```bash
uv run ruff check .
uv run ruff format .
```

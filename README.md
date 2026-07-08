# holder-boost

**holder-boost** — сервис предсказания исхода бинарного маркета Polymarket (**Yes** / **No**)
с помощью CatBoost по поведению холдеров.

Модель обучается на resolved-маркетах: для каждого дня до экспирации (`days_left`) и набора
`proxy_wallet` строится вектор признаков из распределения позиций холдеров по исходам.
Целевая переменная — **`resolved_outcome_index`** (0 = Yes, 1 = No).

**Что делает проект:**

1. **`train`** — загружает resolved-маркеты и holder-distribution из poly-house по **`[train].tag_id`**,
   обучает CatBoost и сохраняет артефакты в **`.models/{tag_id}/`**.
2. **`serve`** — REST API **`POST /predict`**: на вход `tag_id`, `days_left` и текущие позиции холдеров,
   на выход `predict_probs` — `[P(Yes), P(No)]`.

## Принцип работы

Данные для обучения загружаются напрямую из poly-house при каждом запуске **`train`**:
closed-маркеты по тегу (`GET /markets`), список кошельков (`GET /activity/holders`) и
holder-distribution по каждому кошельку (`GET /activity/holder-distribution`).
Обученная модель и метаданные сохраняются в **`.models/{tag_id}/`**
(отдельная папка на каждый тег).

## Требования

- Python 3.12+
- [`uv`](https://github.com/astral-sh/uv)
- Запущенный poly-house на **`poly_house.base_url`** (для **`train`**)

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
- `[poly_house]` — `base_url` (по умолчанию `http://127.0.0.1:8000`); **`train`**
- `[train]` — обучение CatBoost (предсказание **`resolved_outcome_index`**, 0 = Yes, 1 = No):
  - `tag_id` — числовой tag id; JSON body на **`GET /markets`**
  - `model_dir` — базовый каталог артефактов (по умолчанию `.models/`); модель тега — **`{model_dir}/{tag_id}/`**
  - `neg_risk` — query **`neg_risk`** на **`GET /markets`**
  - `min_wallet_count` — минимум вхождений `proxy_wallet` в маркетах для колонки с адресом кошелька
  - `n_trials`, `n_folds`, `random_state` — Optuna и CV
- `[api]` — REST API (**`serve`**):
  - `host`, `port` — адрес сервера (по умолчанию `127.0.0.1:8080`); модели из **`[train].model_dir/{tag_id}/`**

## Команды

### `train`

Один вызов без аргументов (тег из **`[train].tag_id`**):

1. Загрузка closed-маркетов через poly-house **`GET /markets`** (offset-пагинация, `limit` до 500);
   body = `[tag_id]` из **`[train]`**; query **`neg_risk`** из **`[train]`**
2. Для каждого маркета — **`GET /activity/holders`**, затем **`GET /activity/holder-distribution`**
   на каждый `proxy_wallet`; `max_days_left` маркета = max `max_days_left` холдеров (где `>= 0`), иначе `-1` (неизвестно)
3. Разворачивание строк `(market, day)` для `day = 0..max_days_left` (маркеты с `max_days_left < 0` пропускаются); фичи: **`days_left`** и адрес кошелька (`distribution[day]`, иначе `-1`); строки, где все wallet-колонки `-1`, пропускаются
4. Label: **`resolved_outcome_index`**
5. Optuna + 5-fold CV (или **`--skip-tuning`** для быстрого прогона)
6. Сохранение в **`{model_dir}/{tag_id}/`**: `catboost.cbm`, `wallets.pkl`, `metadata.json`, `feature_importance.csv`

При ошибке HTTP poly-house обучение завершается кодом **1**. Ошибка holders/distribution
для отдельного маркета — warning и skip; ошибка одного кошелька — warning и skip кошелька.

```bash
uv run holder-boost train
uv run holder-boost train --skip-tuning
uv run holder-boost train --max-pages 2  # ограничить пагинацию /markets (для тестов)
```

### `serve`

FastAPI-сервер с **`POST /predict`** и **`GET /features/{tag_id}`** (требует обученную модель в **`.models/{tag_id}/`**).

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

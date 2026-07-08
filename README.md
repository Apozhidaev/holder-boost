# holder-boost

**holder-boost** — инструмент для поиска и анализа событий Polymarket, где стратегия
«покупать **No**» может иметь устойчивое преимущество.

Идея взята из статьи [Buying NO for (geo)politics on Polymarket – why it works](https://kurguzkin.com/buying-no-outcomes-for-geopolitics-on-polymarket-why-it-works/):
на политических и геополитических рынках участники переоценивают **Yes** на ярких
сценариях в фокусе медиа — из‑за этого **No** часто торгуется **дешевле** справедливой
цены. Построить надёжную модель для таких событий сложно, поэтому искажение держится.
Отсюда повторяемый паттерн: покупка **No** в диапазоне вроде 80–95¢ с удержанием
до экспирации.

**Цели проекта:**

1. **Скрининг** — находить события с исходом **No** в настраиваемом ценовом диапазоне
   (категории, теги, горизонт до дедлайна и другие фильтры).
2. **История** — искать закономерности на исторических данных: какие метки, диапазоны цен
   и условия дают стабильный результат, а где edge исчезает.

Сейчас в репозитории — синхронизация каталога (**`sync`**), обучение **`train`** и REST API (**`serve`**).

## Принцип работы

Каталог resolved-маркетов живёт в Mongo (**`markets`**): документ по Gamma **`id`**
(`ResolvedMarket`: `condition_id`, `resolved_outcome_index`, `trade_cent_prices`, `stop_prices`, `tags` и т.д.).
Наполняется командой **`sync`** (см. раздел **`sync`** ниже; при необходимости запускайте повторно).

## Требования

- Python 3.12+
- [`uv`](https://github.com/astral-sh/uv)
- Доступный экземпляр MongoDB

## Установка

```bash
uv sync
uv sync --group ml   # для train и serve (CatBoost, FastAPI)
```

Создаётся виртуальное окружение проекта и ставятся runtime- и dev-зависимости
из `pyproject.toml`. Группа **`ml`** нужна для **`train`** и **`serve`**.

## Конфигурация

Настройки лежат в `config.toml` в корне репозитория.

Секции:

- `[mongo]` — `uri`, `db`
- `[http]` — `timeout_seconds`
- `[logging]` — `level`, `format` (`json` / `pretty`), `mongo_ttl_days`
- `[poly_house]` — `base_url` (по умолчанию `http://127.0.0.1:8000`); **`sync`** и trade-цены
- `[sync]` — загрузка resolved-маркетов в **`markets`** из poly-house:
  - `tag_ids` — числовые tag id (`list[int]`); JSON body на **`GET /markets`**; пустой список — без фильтра по тегам
  - `neg_risk` — query **`neg_risk`** на **`GET /markets`**; по умолчанию **`false`**
- `[train]` — обучение CatBoost на holder-distribution из **`markets`** (предсказание **`resolved_outcome_index`**, 0 = Yes, 1 = No):
  - `model_dir` — каталог артефактов (по умолчанию `.models/`)
  - `neg_risk` — фильтр маркетов при обучении
  - `min_wallet_count` — минимум вхождений `proxy_wallet` в маркетах для колонки `proxy_wallet_*`
  - `n_trials`, `n_folds`, `random_state` — Optuna и CV
- `[api]` — REST API (**`serve`**):
  - `host`, `port` — адрес сервера (по умолчанию `127.0.0.1:8080`); модель из **`[train].model_dir`**

## Команды

### `sync`

Один вызов без аргументов:

1. Загрузка closed-маркетов через poly-house **`GET /markets`** (offset-пагинация, `limit` до 500);
   query **`neg_risk`** из **`[sync]`**; опционально body **`tag_ids`** из **`[sync]`**; теги — **`GET /tags/{id}`** для id из маркетов;
   парсинг в **`ResolvedMarket`**.
2. Для **новых** маркетов (нет документа по **`id`**) — параллельно:
   - **`GET /trades/take-prices`** (BUY, `outcome_index` 0 и 1): take-цены в долларах конвертируются в центы и сохраняются в **`trade_cent_prices`** (`[[yes_cents...], [no_cents...]]`);
   - **`GET /trades/stop-prices`** (BUY, `outcome_index = resolved_outcome_index`): агрегаты stop-loss сохраняются в **`stop_prices`** — `dict[cent_price, {count, min_price, min_mean}]` (все значения в центах);
   - **`GET /activity/holder-distribution`**: распределение холдеров по исходам сохраняется в **`holders`** — `[{proxy_wallet, distribution: {outcome_index: count}}]`; **`max_days_left`** — горизонт в днях из того же ответа.
   Требуется запущенный сервис на **`poly_house.base_url`** (по умолчанию `http://127.0.0.1:8000`).
3. **Insert** в коллекцию **`markets`** по **`id`** (уже существующие документы не перезаписываются;
   повторный **`sync`** не backfill'ит trade-, stop-цены и holders).

При ошибке HTTP poly-house синхронизация завершается кодом **1** (stderr и лог). Битые строки
парсинга пропускаются с warning, не валят весь прогон. Требуется запущенный poly-house на **`poly_house.base_url`**.

```bash
uv run holder-boost sync
uv run holder-boost sync --max-pages 2  # ограничить пагинацию /markets (для тестов)
```

### `train`

Один вызов (только Mongo, без HTTP; требует `uv sync --group ml`):

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

FastAPI-сервер с **`POST /predict`** (требует `uv sync --group ml` и обученную модель в **`[train].model_dir`**).

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
curl http://127.0.0.1:8080/health
curl -X POST http://127.0.0.1:8080/predict \
  -H 'Content-Type: application/json' \
  -d '{"days_left": 1, "holders": [{"proxy_wallet": "0x...", "outcome_index": 1}]}'
```

OpenAPI: `http://127.0.0.1:8080/docs`

## Линтинг

```bash
uv run ruff check .
uv run ruff format .
```

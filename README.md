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

Сейчас в репозитории — синхронизация каталога (**`sync`**), расчёт **`predict`** и обучение **`train`**.

## Принцип работы

Каталог resolved-маркетов живёт в Mongo (**`markets`**): документ по Gamma **`id`**
(`ResolvedMarket`: `condition_id`, `resolved_outcome_index`, `trade_cent_prices`, `stop_prices`, `tags` и т.д.).
Наполняется командой **`sync`** (см. раздел **`sync`** ниже; при необходимости запускайте повторно).

Вероятности Yes/No по тегам — в **`predicts`** (документ по **`tag_id`**, поля `outcome_probs`
`[prob_yes, prob_no]`, `trade_price_probs` (`[yes, no]` по ценам), `trade_price_probs_raw`, `yes_win_markets`, `total_markets`). Считаются командой **`predict`** поверх данных **`sync`**.

## Требования

- Python 3.12+
- [`uv`](https://github.com/astral-sh/uv)
- Доступный экземпляр MongoDB

## Установка

```bash
uv sync
uv sync --group ml   # для train (CatBoost)
```

Создаётся виртуальное окружение проекта и ставятся runtime- и dev-зависимости
из `pyproject.toml`. Группа **`ml`** нужна для **`train`**.

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
- `[predict]` — расчёт **`predicts`** из **`markets`**:
  - `neg_risk` — **обязательный**; в расчёт попадают только маркеты с совпадающим **`neg_risk`**
- `[train]` — обучение CatBoost на take-ценах из **`markets`** (оба исхода: **0** = Yes, **1** = No):
  - `model_dir` — корень артефактов (по умолчанию `models/`); модели в **`outcome_0/`** и **`outcome_1/`**
  - `neg_risk` — фильтр маркетов при обучении
  - `min_tag_count` — минимум вхождений тега в датасете для multi-hot
  - `n_trials`, `n_folds`, `random_state` — Optuna и CV

## Команды

### `sync`

Один вызов без аргументов:

1. Загрузка closed-маркетов через poly-house **`GET /markets`** (offset-пагинация, `limit` до 500);
   query **`neg_risk`** из **`[sync]`**; опционально body **`tag_ids`** из **`[sync]`**; теги — **`GET /tags/{id}`** для id из маркетов;
   парсинг в **`ResolvedMarket`**.
2. Для **новых** маркетов (нет документа по **`id`**) — параллельно:
   - **`GET /trades/take-prices`** (BUY, `outcome_index` 0 и 1): take-цены в долларах конвертируются в центы и сохраняются в **`trade_cent_prices`** (`[[yes_cents...], [no_cents...]]`);
   - **`GET /trades/stop-prices`** (BUY, `outcome_index = resolved_outcome_index`): агрегаты stop-loss сохраняются в **`stop_prices`** — `dict[cent_price, {count, min_price, min_mean}]` (все значения в центах);
   - **`GET /activity/holder-distribution`**: распределение холдеров по исходам сохраняется в **`holders`** — `[{proxy_wallet, distribution: {outcome_index: count}}]`.
   Требуется запущенный сервис на **`poly_house.base_url`** (по умолчанию `http://127.0.0.1:8000`).
3. **Insert** в коллекцию **`markets`** по **`id`** (уже существующие документы не перезаписываются;
   повторный **`sync`** не backfill'ит trade-, stop-цены и holders).

При ошибке HTTP poly-house синхронизация завершается кодом **1** (stderr и лог). Битые строки
парсинга пропускаются с warning, не валят весь прогон. Требуется запущенный poly-house на **`poly_house.base_url`**.

```bash
uv run holder-boost sync
uv run holder-boost sync --max-pages 2  # ограничить пагинацию /markets (для тестов)
```

### `predict`

Один вызов без аргументов (только Mongo, без HTTP):

1. Загрузка всех тегов из **`tags`**
2. Для каждого тега — маркеты из **`markets`**, где `tag_id ∈ tags[]` и **`neg_risk`** совпадает с **`[predict].neg_risk`**
3. Расчёт **`outcome_probs = [yes_win_markets / total_markets, 1 - prob_yes]`**
   (индекс **0** = Yes, **1** = No), где **`yes_win_markets`** — маркеты с **`resolved_outcome_index == 0`**
4. Расчёт **`trade_price_probs_raw`** — по каждому исходу 101 значение (цены **0–100**¢, `float` или `null`) до импутации;
   **`trade_price_probs[outcome_index]`** — ряд после заполнения разрывов. Для каждой цены `p` доля take-цен `p` в
   **`trade_cent_prices[outcome_index]`** маркетов с **`resolved_outcome_index == outcome_index`** от всех наблюдений `p`
   по тегу. Импутация: **0¢ → 0.0** если нет наблюдений, каждый разрыв между **соседними** измеренными точками —
   линейная интерполяция; **100¢** и хвост после последней измеренной — значение последней измеренной точки.
   Пустые trade-цены не влияют на счётчики, но маркет учитывается в **`outcome_probs`** / **`total_markets`**.
5. **Upsert** в **`predicts`** по **`tag_id`**; теги без маркетов пропускаются

Требует предварительного **`sync`**.

```bash
uv run holder-boost predict
```

### `train`

Один вызов (только Mongo, без HTTP; требует `uv sync --group ml`):

1. Загрузка всех маркетов из **`markets`** с **`neg_risk`** из **`[train]`**
2. Для каждого исхода (**0** = Yes, **1** = No): разворачивание **`trade_cent_prices[outcome_index]`**; label **`won = 1`**, если **`resolved_outcome_index == outcome_index`**
3. Multi-hot частых тегов + признак **`price`** (0–100¢)
4. Optuna + 5-fold CV (или **`--skip-tuning`** для быстрого прогона)
5. Сохранение в **`model_dir/outcome_{0,1}/`**: `catboost.cbm`, `mlb.pkl`, `metadata.json`, `feature_importance.csv`

Требует предварительного **`sync`**.

```bash
uv run holder-boost train
uv run holder-boost train --skip-tuning   # без Optuna
```

## Линтинг

```bash
uv run ruff check .
uv run ruff format .
```

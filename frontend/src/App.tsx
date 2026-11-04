import { useEffect, useMemo, useState, type FormEvent } from 'react'
import {
  Activity,
  ArrowDownToLine,
  ArrowUpRight,
  Boxes,
  CalendarDays,
  Check,
  ChevronDown,
  CircleHelp,
  Clock3,
  LayoutDashboard,
  Leaf,
  Menu,
  PackageCheck,
  PanelLeftClose,
  RotateCcw,
  Search,
  Table2,
} from 'lucide-react'
import './App.css'
import { demandMetrics } from './demandMetrics'
import { itemName, storeName } from './catalog'
import ForecastChart, { type ChartPoint } from './ForecastChart'

const API_BASE = (import.meta.env.VITE_API_BASE_URL ?? 'http://127.0.0.1:8000').replace(/\/$/, '')

type PageKey = 'forecast' | 'sales' | 'models' | 'inventory'
type ViewMode = 'chart' | 'table' | 'split'
type Status = 'connecting' | 'online' | 'offline'

interface SalesOptions {
  stores: number[]
  items: number[]
  date_start: string
  date_end: string
}

interface SalesRecord {
  date: string
  store: number
  item: number
  sales: number
}

interface SalesSummary {
  records: number
  total_sales: number
  average_daily_sales_per_record: number | null
  date_start: string | null
  date_end: string | null
}

interface ForecastRecord {
  ds?: string
  date?: string
  yhat?: number
  prediction?: number
  predicted_sales?: number
  forecast?: number
  yhat_lower?: number
  yhat_upper?: number
  _source?: string
}

interface ModelMetric {
  Model: string
  MAE?: number
  RMSE?: number
  'MAPE (%)'?: number
  'sMAPE (%)'?: number
}

interface TrainingReport {
  store: number
  item: number
  report: {
    best_model?: string
    holdout_days?: number
    comparison?: ModelMetric[]
  }
}

interface ComparisonResponse {
  training_reports: TrainingReport[]
}

interface HistoricalComparisonResponse {
  records: Array<{
    date: string
    actual_sales: number
    yhat: number
    yhat_lower: number
    yhat_upper: number
  }>
}

interface ForecastResponse {
  available: boolean
  records: ForecastRecord[]
}

interface InventoryResponse {
  forecast_model: string
  recommendation: Record<string, string | number | boolean>
  note: string
}

const pages: { id: PageKey; label: string; icon: typeof LayoutDashboard; hint: string }[] = [
  { id: 'forecast', label: 'Forecast', icon: Activity, hint: 'Demand outlook' },
  { id: 'sales', label: 'Sales history', icon: LayoutDashboard, hint: 'Daily performance' },
  { id: 'models', label: 'Model comparison', icon: Boxes, hint: 'Holdout metrics' },
  { id: 'inventory', label: 'Inventory', icon: PackageCheck, hint: 'Reorder planning' },
]

async function getJson<T>(path: string, signal?: AbortSignal): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, { signal, cache: 'no-store' })
  const payload = await response.json().catch(() => null) as { detail?: string } | null
  if (!response.ok) {
    throw new Error(payload?.detail ?? `Request failed (${response.status})`)
  }
  return payload as T
}

function queryString(params: Record<string, string | number | undefined>): string {
  const query = new URLSearchParams()
  Object.entries(params).forEach(([key, value]) => {
    if (value !== undefined) query.set(key, String(value))
  })
  return query.toString()
}

function recordDate(record: ForecastRecord): string {
  return record.ds ?? record.date ?? ''
}

function recordForecast(record: ForecastRecord): number {
  return Number(record.yhat ?? record.prediction ?? record.predicted_sales ?? record.forecast ?? NaN)
}

function formatNumber(value: number | null | undefined, digits = 0): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return '—'
  return new Intl.NumberFormat('en-US', { maximumFractionDigits: digits }).format(value)
}

function formatDate(value: string | null | undefined): string {
  if (!value) return '—'
  const date = new Date(`${value.slice(0, 10)}T00:00:00`)
  return Number.isNaN(date.getTime())
    ? value
    : new Intl.DateTimeFormat('en-US', { month: 'short', day: 'numeric', year: 'numeric' }).format(date)
}

function compactDate(value: string): string {
  const date = new Date(`${value.slice(0, 10)}T00:00:00`)
  return Number.isNaN(date.getTime())
    ? value
    : new Intl.DateTimeFormat('en-US', { month: 'short', day: 'numeric' }).format(date)
}

function metricValue(value: number | undefined): string {
  return value === undefined ? '—' : formatNumber(value, 2)
}

function App() {
  const [page, setPage] = useState<PageKey>('forecast')
  const [menuOpen, setMenuOpen] = useState(false)
  const [status, setStatus] = useState<Status>('connecting')
  const [options, setOptions] = useState<SalesOptions | null>(null)
  const [dateDraft, setDateDraft] = useState({ start: '', end: '' })
  const [dateRange, setDateRange] = useState({ start: '', end: '' })
  const [refreshKey, setRefreshKey] = useState(0)
  const [inventoryRefreshKey, setInventoryRefreshKey] = useState(0)
  const [itemsStore, setItemsStore] = useState<number | null>(null)
  const [store, setStore] = useState<number | null>(null)
  const [item, setItem] = useState<number | null>(null)
  const [items, setItems] = useState<number[]>([])
  const [horizon, setHorizon] = useState(30)
  const [model, setModel] = useState('xgboost')
  const [viewMode, setViewMode] = useState<ViewMode>('chart')
  const [sales, setSales] = useState<SalesRecord[]>([])
  const [summary, setSummary] = useState<SalesSummary | null>(null)
  const [forecasts, setForecasts] = useState<ForecastRecord[]>([])
  const [historicalComparison, setHistoricalComparison] = useState<HistoricalComparisonResponse['records']>([])
  const [trainingReport, setTrainingReport] = useState<TrainingReport | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [currentStock, setCurrentStock] = useState('')
  const [leadTime, setLeadTime] = useState(7)
  const [cycleDays, setCycleDays] = useState(14)
  const [inventory, setInventory] = useState<InventoryResponse | null>(null)
  const [inventoryLoading, setInventoryLoading] = useState(false)
  const [inventoryError, setInventoryError] = useState('')
  const [inventoryRequest, setInventoryRequest] = useState<{ stock: number; leadTime: number; cycleDays: number } | null>(null)

  useEffect(() => {
    let active = true
    Promise.all([getJson<{ status: string }>('/health'), getJson<SalesOptions>('/sales/options')])
      .then(([, data]) => {
        if (!active) return
        setOptions(data)
        setDateDraft({ start: data.date_start, end: data.date_end })
        setDateRange({ start: data.date_start, end: data.date_end })
        setItems(data.items)
        setStore(data.stores[0] ?? null)
        setItem(data.items[0] ?? null)
        setStatus('online')
        if (!data.stores.length || !data.items.length) setLoading(false)
      })
      .catch((cause: unknown) => {
        if (!active) return
        setStatus('offline')
        setLoading(false)
        setError(cause instanceof Error ? cause.message : 'Unable to connect to the RetailLens API.')
      })
    return () => { active = false }
  }, [])

  useEffect(() => {
    if (store === null) return
    let active = true
    getJson<SalesOptions>(`/sales/options?${queryString({ store })}`)
      .then((data) => {
        if (!active) return
        setItems(data.items)
        setItem((current) => current !== null && data.items.includes(current) ? current : data.items[0] ?? null)
        setItemsStore(store)
      })
      .catch((cause: unknown) => {
        if (active) {
          setError(cause instanceof Error ? cause.message : 'Unable to load item choices.')
          setLoading(false)
        }
      })
    return () => { active = false }
  }, [store])

  useEffect(() => {
    setSummary(null)
    setSales([])
    setForecasts([])
    setHistoricalComparison([])
    setTrainingReport(null)
    setError('')
    if (store === null || item === null || itemsStore !== store) {
      setLoading(store !== null && itemsStore !== store)
      return
    }
    if (dateRange.start && dateRange.end && dateRange.start > dateRange.end) {
      setError('The start date must be on or before the end date.')
      setLoading(false)
      return
    }
    setLoading(true)
    let active = true
    const controller = new AbortController()
    const filter = queryString({
      store,
      item,
      start_date: dateRange.start || undefined,
      end_date: dateRange.end || undefined,
    })
    Promise.all([
      getJson<SalesSummary>(`/sales/summary?${filter}`, controller.signal),
      getJson<{ records: SalesRecord[] }>(`/sales?${queryString({
        store,
        item,
        start_date: dateRange.start || undefined,
        end_date: dateRange.end || undefined,
        limit: 10000,
      })}`, controller.signal),
      getJson<ForecastResponse>(`/forecasts?${queryString({ store, item, limit: 1000 })}`, controller.signal),
      getJson<HistoricalComparisonResponse>(`/sales/comparison?${filter}`, controller.signal),
      getJson<ComparisonResponse>(`/models/comparison?${queryString({ store, item, holdout_days: horizon })}`, controller.signal),
    ])
      .then(([salesSummary, salesPage, forecastPage, comparisonPage, modelData]) => {
        if (!active) return
        setSummary(salesSummary)
        setSales(salesPage.records)
        setForecasts(forecastPage.records)
        setHistoricalComparison(comparisonPage.records)
        setTrainingReport(modelData.training_reports[0] ?? null)
      })
      .catch((cause: unknown) => {
        if (!active) return
        setError(cause instanceof Error ? cause.message : 'Unable to load this store-item series.')
        setSummary(null)
        setSales([])
        setForecasts([])
        setHistoricalComparison([])
        setTrainingReport(null)
      })
      .finally(() => { if (active) setLoading(false) })
    return () => { active = false; controller.abort() }
  }, [store, item, itemsStore, dateRange.start, dateRange.end, refreshKey, horizon])

  const bestModel = trainingReport?.report.best_model ?? ''
  const selectedModelName = model === 'xgboost' ? 'XGBoost' : 'LSTM'
  const visibleForecasts = useMemo(() => {
    const normalized = selectedModelName.toLowerCase()
    return forecasts
      .filter((record) => Number.isFinite(recordForecast(record)))
      .filter((record) => !normalized || (record._source ?? '').toLowerCase().includes(normalized.split(' ')[0]))
      .sort((left, right) => recordDate(left).localeCompare(recordDate(right)))
      .slice(0, horizon)
  }, [forecasts, horizon, model, selectedModelName])
  // Keep the future horizon readable instead of squeezing it beside years of history.
  // The sales history table and summary still use the entire selected date range.
  const historical = useMemo(() => sales.slice(-90), [sales])
  const chartData = useMemo<ChartPoint[]>(() => [
    ...historical.map((record) => ({
      date: record.date,
      actual: record.sales,
      forecast: null,
    })),
    ...visibleForecasts.map((record) => ({
      date: recordDate(record),
      actual: null,
      forecast: recordForecast(record),
      confidence: record.yhat_lower != null && record.yhat_upper != null
        ? [Number(record.yhat_lower), Number(record.yhat_upper)] as [number, number]
        : undefined,
    })),
  ], [historical, visibleForecasts])
  const tableData = useMemo<ChartPoint[]>(() => [
    ...chartData.filter((record) => record.forecast !== null),
    ...sales.map((record) => {
      const comparison = historicalComparison.find((row) => row.date === record.date)
      return {
        date: record.date,
        actual: record.sales,
        forecast: comparison?.yhat ?? null,
        confidence: comparison
          ? [comparison.yhat_lower, comparison.yhat_upper] as [number, number]
          : undefined,
      }
    }),
  ], [chartData, historicalComparison, sales])
  const metrics = demandMetrics(sales, visibleForecasts.map(recordForecast), horizon)
  const salesWindow = useMemo(() => {
    const records = sales
    const total = records.reduce((sum, record) => sum + record.sales, 0)
    return {
      total,
      average: records.length ? total / records.length : null,
      observations: records.length,
    }
  }, [sales, horizon])
  const forecastUnavailable = !loading && visibleForecasts.length === 0
  const comparison = trainingReport?.report.comparison ?? []
  const stockValue = Number(currentStock)

  // Refresh only model artifacts so actual sales remain visible while training runs.
  useEffect(() => {
    if (!forecastUnavailable || error || store === null || item === null) return
    const controller = new AbortController()
    let active = true
    let timer: number
    async function checkForecasts() {
      try {
        const filter = queryString({ store: store!, item: item!, limit: 1000 })
        const [forecastPage, modelData] = await Promise.all([
          getJson<ForecastResponse>(`/forecasts?${filter}`, controller.signal),
          getJson<ComparisonResponse>(`/models/comparison?${queryString({ store: store!, item: item!, holdout_days: horizon })}`, controller.signal),
        ])
        if (!active) return
        setForecasts(forecastPage.records)
        setTrainingReport(modelData.training_reports[0] ?? null)
      } catch {
        // Keep already loaded sales visible and retry temporary connection failures.
      } finally {
        if (active) timer = window.setTimeout(checkForecasts, 5000)
      }
    }
    timer = window.setTimeout(checkForecasts, 5000)
    return () => { active = false; window.clearTimeout(timer); controller.abort() }
  }, [forecastUnavailable, error, store, item, model, horizon])

  function changeStore(value: number) {
    setLoading(true)
    setError('')
    setInventory(null)
    setInventoryRequest(null)
    setStore(value)
  }

  function changeItem(value: number) {
    setLoading(true)
    setError('')
    setInventory(null)
    setInventoryRequest(null)
    setItem(value)
  }

  function applyDateSearch(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    if (dateDraft.start && dateDraft.end && dateDraft.start > dateDraft.end) {
      setError('The start date must be on or before the end date.')
      return
    }
    setDateRange(dateDraft)
    setRefreshKey((current) => current + 1)
  }

  function resetDateSearch() {
    if (!options) return
    const fullRange = { start: options.date_start, end: options.date_end }
    setDateDraft(fullRange)
    setDateRange(fullRange)
    setRefreshKey((current) => current + 1)
  }

  function calculateInventory(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    if (currentStock.trim() === '' || !Number.isFinite(stockValue) || stockValue < 0
      || !Number.isInteger(leadTime) || leadTime < 1
      || !Number.isInteger(cycleDays) || cycleDays < 1) {
      setInventoryError('Enter a non-negative stock value and positive whole days for lead time and order cycle.')
      return
    }
    setInventoryError('')
    setInventoryRequest({ stock: stockValue, leadTime, cycleDays })
    setInventoryRefreshKey((current) => current + 1)
  }

  useEffect(() => {
    setInventory(null)
    setInventoryError('')
    setInventoryLoading(false)
    if (page !== 'inventory' || store === null || item === null || itemsStore !== store
      || inventoryRequest === null) return
    const controller = new AbortController()
    let active = true
    setInventoryLoading(true)
    const timer = window.setTimeout(() => {
      const params = queryString({
        store, item, model,
        current_stock: inventoryRequest.stock,
        lead_time_days: inventoryRequest.leadTime,
        cycle_stock_days: inventoryRequest.cycleDays,
      })
      getJson<InventoryResponse>(`/inventory/recommendation?${params}`, controller.signal)
        .then((data) => { if (active) setInventory(data) })
        .catch((cause: unknown) => {
          if (active) setInventoryError(cause instanceof Error ? cause.message : 'Unable to calculate inventory guidance.')
        })
        .finally(() => { if (active) setInventoryLoading(false) })
    }, 300)
    return () => { active = false; window.clearTimeout(timer); controller.abort() }
  }, [page, store, item, itemsStore, model, inventoryRequest, inventoryRefreshKey])

  function downloadForecast() {
    const rows = [
      ['date', 'forecast', 'lower_bound', 'upper_bound'],
      ...visibleForecasts.map((record) => [
        recordDate(record),
        String(recordForecast(record)),
        record.yhat_lower === undefined ? '' : String(record.yhat_lower),
        record.yhat_upper === undefined ? '' : String(record.yhat_upper),
      ]),
    ]
    const csv = rows.map((row) => row.map((field) => `"${field.replaceAll('"', '""')}"`).join(',')).join('\n')
    const link = document.createElement('a')
    link.href = URL.createObjectURL(new Blob([csv], { type: 'text/csv;charset=utf-8' }))
    link.download = `retail-lens-store-${store}-item-${item}-forecast.csv`
    link.click()
    window.setTimeout(() => URL.revokeObjectURL(link.href), 0)
  }

  const navigation = (
    <nav className="drawer-nav" aria-label="Main features">
      <p className="drawer-section-label">Workspace</p>
      {pages.map(({ id, label, icon: Icon, hint }) => (
        <button
          className={`drawer-link ${page === id ? 'is-active' : ''}`}
          key={id}
          onClick={() => { setPage(id); setMenuOpen(false) }}
          type="button"
        >
          <span className="drawer-icon"><Icon size={18} strokeWidth={1.8} /></span>
          <span className="drawer-link-copy">
            <strong>{label}</strong>
            <small>{hint}</small>
          </span>
          {page === id && <span className="active-dot" />}
        </button>
      ))}
    </nav>
  )

  return (
    <div className="app-shell">
      {menuOpen && <button className="drawer-backdrop" aria-label="Close menu" onClick={() => setMenuOpen(false)} />}
      <aside className={`feature-drawer ${menuOpen ? 'drawer-open' : ''}`} aria-hidden={!menuOpen} inert={!menuOpen}>
        <div className="drawer-brand">
          <span className="brand-mark"><Leaf size={19} fill="currentColor" /></span>
          <span><strong>RetailLens</strong><small>Demand intelligence</small></span>
          <button className="icon-button drawer-close" aria-label="Close menu" onClick={() => setMenuOpen(false)} type="button">
            <PanelLeftClose size={18} />
          </button>
        </div>
        <div className="drawer-context">
          <span className="context-orb"><Activity size={17} /></span>
          <span><strong>Forecast workspace</strong><small>Store and item insights</small></span>
        </div>
        {navigation}
        <div className="drawer-bottom">
          <span className={`connection-pip ${status}`} />
          <span><strong>{status === 'online' ? 'API connected' : status === 'offline' ? 'API unavailable' : 'Connecting…'}</strong><small>RetailLens local service</small></span>
        </div>
      </aside>

      <div className="main-column">
        <header className="topbar">
          <div className="topbar-left">
            <button className="menu-toggle" onClick={() => setMenuOpen(true)} type="button" aria-label="Open feature menu" aria-expanded={menuOpen}>
              <Menu size={19} />
            </button>
            <span className="breadcrumb">RetailLens <span>/</span> {pages.find((entry) => entry.id === page)?.label}</span>
          </div>
          <div className="topbar-right">
            <span className={`api-badge ${status}`}><span className="connection-pip" />{status === 'online' ? 'Live data' : status === 'offline' ? 'Offline' : 'Connecting'}</span>
            <span className="topbar-date">{formatDate(options?.date_end)}</span>
            <span className="avatar">RL</span>
          </div>
        </header>

        <main className="page-content">
          <div className="page-heading">
            <div>
              <p className="eyebrow"><span className="eyebrow-line" />RETAIL DEMAND INTELLIGENCE</p>
              <h1>{page === 'forecast' ? 'Demand forecast' : pages.find((entry) => entry.id === page)?.label}</h1>
              <p className="page-subtitle">A clearer view of what your shelves may need next.</p>
            </div>
            <div className="heading-date">
              <span className="date-icon"><Clock3 size={16} /></span>
              <span><small>Historical period</small><strong>{formatDate(summary?.date_start ?? dateRange.start ?? options?.date_start)} – {formatDate(summary?.date_end ?? dateRange.end ?? options?.date_end)}</strong></span>
            </div>
          </div>

          <p className="panel-subtitle">Store and product names are demo aliases; the source dataset contains numeric IDs only.</p>
          <section className="control-panel" aria-label="Forecast controls">
            <div className="selector-group">
              <label className="field-label" htmlFor="store-select">Store</label>
              <div className="select-wrap">
                <select id="store-select" value={store ?? ''} onChange={(event) => changeStore(Number(event.target.value))} disabled={!options}>
                  {(options?.stores ?? []).map((value) => <option key={value} value={value}>{storeName(value)}</option>)}
                </select>
                <ChevronDown size={15} />
              </div>
            </div>
            <span className="control-divider" />
            <div className="selector-group">
              <label className="field-label" htmlFor="item-select">Product</label>
              <div className="select-wrap">
                <select id="item-select" value={item ?? ''} onChange={(event) => changeItem(Number(event.target.value))} disabled={!items.length || itemsStore !== store}>
                  {items.map((value) => <option key={value} value={value}>{itemName(value)}</option>)}
                </select>
                <ChevronDown size={15} />
              </div>
            </div>
            {page !== 'models' && <>
              <span className="control-divider" />
              <div className="selector-group model-selector">
                <label className="field-label" htmlFor="model-select">Forecast model</label>
                <div className="select-wrap">
                  <select id="model-select" value={model} onChange={(event) => { setModel(event.target.value); setInventory(null); setInventoryRequest(null) }}>
                    <option value="lstm">LSTM</option>
                    <option value="xgboost">XGBoost</option>
                  </select>
                  <ChevronDown size={15} />
                </div>
              </div>
            </>}
            {(page === 'forecast' || page === 'models') && (
              <div className="horizon-picker" aria-label="Forecast horizon">
                {[30, 60, 90].map((days) => (
                  <button className={horizon === days ? 'selected' : ''} key={days} onClick={() => setHorizon(days)} aria-pressed={horizon === days} type="button">
                    {days}<span>d</span>
                  </button>
                ))}
              </div>
            )}
          </section>
          {(page === 'forecast' || page === 'sales') && (
            <form className="date-search-panel" onSubmit={applyDateSearch}>
              <div className="date-search-title">
                <span className="date-search-icon"><CalendarDays size={16} /></span>
                <span><strong>Search by date</strong><small>Dates update automatically. Future forecast dates are shown separately.</small></span>
              </div>
              <label className="date-field" htmlFor="start-date">
                <span>From</span>
                <input
                  id="start-date"
                  type="date"
                  min={options?.date_start}
                  max={dateDraft.end || options?.date_end}
                  value={dateDraft.start}
                  onChange={(event) => setDateDraft((current) => ({ ...current, start: event.target.value }))}
                  disabled={!options}
                />
              </label>
              <label className="date-field" htmlFor="end-date">
                <span>To</span>
                <input
                  id="end-date"
                  type="date"
                  min={dateDraft.start || options?.date_start}
                  max={options?.date_end}
                  value={dateDraft.end}
                  onChange={(event) => setDateDraft((current) => ({ ...current, end: event.target.value }))}
                  disabled={!options}
                />
              </label>
              <button className="date-search-button" disabled={!options || loading} type="submit">
                <Search size={15} />Search
              </button>
              <button className="date-reset-button" disabled={!options || loading} onClick={resetDateSearch} type="button" aria-label="Reset date range">
                <RotateCcw size={15} /><span>Reset</span>
              </button>
            </form>
          )}

          {error && (
            <div className="notice notice-error" role="alert">
              <CircleHelp size={18} />
              <span><strong>Data could not be loaded</strong><small>{error}. Check that the API is running and sales data is available.</small></span>
            </div>
          )}

          {page === 'forecast' && (
            <>
              <section className="metric-grid" aria-label="Series summary">
                <article className="metric-card">
                  <span className="metric-icon green"><Activity size={17} /></span>
                  <span className="metric-label">Historical sales</span>
                  <strong className="metric-value">{loading ? '…' : formatNumber(summary?.total_sales)}</strong>
                  <small className="metric-foot">{formatNumber(summary?.records)} daily records</small>
                </article>
                <article className="metric-card">
                  <span className="metric-icon beige"><ArrowUpRight size={17} /></span>
                  <span className="metric-label">{metrics.isForecast ? 'Forecast demand' : 'Recent sales'}</span>
                  <strong className="metric-value">{loading ? "Loading..." : formatNumber(metrics.total)}</strong>
                  <small className="metric-foot" aria-live="polite">{metrics.isForecast ? `Next ${metrics.days} days - units` : `Actual sales - latest ${metrics.days} days in your date range`}</small>
                </article>
                <article className="metric-card">
                  <span className="metric-icon green-light"><Leaf size={17} /></span>
                  <span className="metric-label">{metrics.isForecast ? 'Daily average' : 'Historical daily average'}</span>
                  <strong className="metric-value">{loading ? "Loading..." : formatNumber(metrics.average, 1)}</strong>
                  <small className="metric-foot" aria-live="polite">{metrics.isForecast ? 'Projected units per day' : 'Actual units per day; forecast will appear when available'}</small>
                </article>
                <article className="metric-card model-card">
                  <span className="metric-icon green"><Check size={17} /></span>
                  <span className="metric-label">Selected model</span>
                  <strong className="metric-value model-name">{selectedModelName || 'Not evaluated'}</strong>
                  <small className="metric-foot">{trainingReport ? `${trainingReport.report.holdout_days ?? 90}-day holdout evaluation` : 'Train this series to compare models'}</small>
                </article>
              </section>

              <section className="forecast-panel">
                <div className="panel-heading">
                  <div>
                    <p className="section-kicker">DEMAND OUTLOOK</p>
                    <h2>Sales trajectory <span className="heading-separator">·</span> {storeName(store)}, {itemName(item)}</h2>
                    <p className="panel-subtitle" aria-live="polite">
                      {selectedModelName} · {horizon}-day forecast · Showing {visibleForecasts.length} forecast days.
                      {' '}Chart: latest {historical.length} historical days. Table: all {sales.length} historical records in your date range.
                    </p>
                  </div>
                  <div className="panel-actions">
                    <div className="view-switch" aria-label="Forecast layout">
                      <button className={viewMode === 'chart' ? 'active' : ''} onClick={() => setViewMode('chart')} title="Chart layout" type="button"><Activity size={16} /><span>Chart</span></button>
                      <button className={viewMode === 'table' ? 'active' : ''} onClick={() => setViewMode('table')} title="Table layout" type="button"><Table2 size={16} /><span>Table</span></button>
                      <button className={viewMode === 'split' ? 'active' : ''} onClick={() => setViewMode('split')} title="Split layout" type="button"><LayoutDashboard size={16} /><span>Split</span></button>
                    </div>
                    <button className="export-button" onClick={downloadForecast} disabled={!visibleForecasts.length} type="button">
                      <ArrowDownToLine size={15} /><span>Export</span>
                    </button>
                  </div>
                </div>
                {!loading && visibleForecasts.length < horizon && (
                  <p className="panel-subtitle" role="status">
                    {visibleForecasts.length === 0
                      ? 'No saved predictions are available for this model and product. Only historical sales can be shown.'
                      : `Only ${visibleForecasts.length} of the requested ${horizon} forecast days are available for this model.`}
                  </p>
                )}
                <div className={`forecast-layout layout-${viewMode}`}>
                  {viewMode !== 'table' && (
                    <div className="chart-area">
                      {loading ? (
                        <div className="loading-state"><span className="spinner" />Loading the selected series…</div>
                      ) : chartData.length ? (
                        <ForecastChart
                          data={chartData}
                          formatDate={formatDate}
                          compactDate={compactDate}
                          formatNumber={formatNumber}
                        />
                      ) : (
                        <div className="empty-chart">
                          <span className="empty-icon"><Activity size={22} /></span>
                          <strong>No future forecast found</strong>
                          <p>Train {storeName(store)}, {itemName(item)} to create forecast files for this view.</p>
                        </div>
                      )}
                    </div>
                  )}
                  {viewMode !== 'chart' && (
                    <div className="forecast-table-wrap">
                      {viewMode === 'split' && <h3>Sales and forecast detail</h3>}
                      <ForecastTable records={tableData} loading={loading} />
                    </div>
                  )}
                </div>
                <div className="chart-footer">
                  <div className="chart-legend-copy"><span className="legend-dot actual-dot" />Actual sales <span className="legend-dot forecast-dot" />Model forecast {visibleForecasts.some((record) => record.yhat_lower != null && record.yhat_upper != null) && <><span className="legend-band" />Uncertainty range</>}</div>
                  <span>Bounds are approximate prediction ranges estimated from holdout errors.</span>
                </div>
              </section>

              <section className="insight-strip">
                <div className="insight-icon"><Leaf size={19} /></div>
                <div><strong>Plan with a little more confidence.</strong><p>Forecasts are estimates from historical sales, not confirmed orders. Check stock availability before acting.</p></div>
                <button onClick={() => setPage('inventory')} type="button">Inventory planning <ArrowUpRight size={15} /></button>
              </section>
            </>
          )}

          {page === 'sales' && (
            <section className="content-card">
              <div className="section-card-heading"><div><p className="section-kicker">HISTORICAL PERFORMANCE</p><h2>Daily sales history</h2><p className="panel-subtitle">{storeName(store)} · {itemName(item)} · {formatDate(summary?.date_start)} – {formatDate(summary?.date_end)}</p></div></div>
              <div className="sales-stat-row"><span><small>Total units</small><strong>{formatNumber(salesWindow.total)}</strong></span><span><small>Daily average</small><strong>{formatNumber(salesWindow.average, 1)}</strong></span><span><small>Observations</small><strong>{formatNumber(salesWindow.observations)}</strong></span></div>
              <SalesTable records={[...sales].reverse()} loading={loading} />
            </section>
          )}

          {page === 'models' && (
            <section className="content-card">
              <div className="section-card-heading"><div><p className="section-kicker">VALIDATION SNAPSHOT</p><h2>Compare forecasting models</h2><p className="panel-subtitle">Chronological holdout results for {storeName(store)}, {itemName(item)}.</p></div><span className="holdout-chip"><Check size={14} />{trainingReport?.report.holdout_days ?? 90} day holdout</span></div>
              {comparison.length ? (
                <div className="comparison-grid">
                  {comparison.map((metric) => (
                    <article className={`comparison-card ${metric.Model === bestModel ? 'winner' : ''}`} key={metric.Model}>
                      <div className="comparison-card-top"><span className="model-icon">{metric.Model.toLowerCase().includes('lstm') ? <Activity size={18} /> : <Boxes size={18} />}</span>{metric.Model === bestModel && <span className="winner-tag"><Check size={13} />Selected</span>}</div>
                      <h3>{metric.Model}</h3>
                      <p>Holdout evaluation</p>
                      <div className="comparison-values"><span><small>MAE</small><strong>{metricValue(metric.MAE)}</strong></span><span><small>RMSE</small><strong>{metricValue(metric.RMSE)}</strong></span><span><small>MAPE</small><strong>{metricValue(metric['MAPE (%)'])}{metric['MAPE (%)'] === undefined ? '' : '%'}</strong></span></div>
                    </article>
                  ))}
                </div>
              ) : (
                <div className="empty-state"><span className="empty-icon"><Boxes size={22} /></span><strong>No model report for this series</strong><p>Train this store-item series before comparing its LSTM and XGBoost results.</p></div>
              )}
              <p className="method-note">The training pipeline selects a model by holdout RMSE, then MAE. Compare models on a separate final test window before using metrics as a production performance claim.</p>
            </section>
          )}

          {page === 'inventory' && (
            <section className="inventory-grid">
              <div className="content-card inventory-form-card">
                <p className="section-kicker">REPLENISHMENT PLANNING</p>
                <h2>Estimate a reorder point</h2>
                <p className="panel-subtitle">Use the selected forecast model and your current on-hand stock. Guidance updates automatically.</p>
                <form className="inventory-form" onSubmit={calculateInventory}>
                  <label>Current stock <span>units on hand</span><input min="0" required step="1" type="number" value={currentStock} onChange={(event) => { setCurrentStock(event.target.value); setInventory(null) }} placeholder="Enter your stock" /></label>
                  <div className="form-duo">
                    <label>Lead time <span>days</span><input min="1" required type="number" value={leadTime} onChange={(event) => { setLeadTime(Number(event.target.value)); setInventory(null) }} /></label>
                    <label>Order cycle <span>days</span><input min="1" required type="number" value={cycleDays} onChange={(event) => { setCycleDays(Number(event.target.value)); setInventory(null) }} /></label>
                  </div>
                  <button className="primary-button" disabled={inventoryLoading || store === null || item === null} type="submit">{inventoryLoading ? 'Calculating…' : 'Calculate recommendation'}<ArrowUpRight size={16} /></button>
                </form>
                {inventoryError && <p className="form-error" role="alert">{inventoryError}</p>}
              </div>
              <div className="content-card recommendation-card">
                <div className="section-card-heading"><div><p className="section-kicker">RECOMMENDATION</p><h2>Stock guidance</h2></div><span className="recommendation-badge"><PackageCheck size={15} />{inventory ? inventory.forecast_model.toUpperCase() : 'READY'}</span></div>
                {inventory ? <InventoryResults inventory={inventory} /> : <div className="recommendation-empty"><span className="recommendation-emblem"><Leaf size={25} /></span><strong>Your estimate will appear here</strong><p>Enter your on-hand stock to calculate lead-time demand, safety stock, and a suggested order quantity.</p></div>}
              </div>
            </section>
          )}

          <footer className="page-footer"><span>RetailLens <span className="footer-dot">·</span> local demand intelligence</span><span>Forecasts inform planning; they do not replace stock verification.</span></footer>
        </main>
      </div>
    </div>
  )
}

function ForecastTable({ records, loading }: { records: ChartPoint[]; loading: boolean }) {
  if (loading) return <div className="loading-state"><span className="spinner" />Loading sales and forecasts…</div>
  if (!records.length) return <div className="table-empty">No sales or forecast rows are available for this selection.</div>
  const actualRecords = records.filter((record) => record.actual !== null)
    .sort((left, right) => right.date.localeCompare(left.date))
  const forecastRecords = records.filter((record) => record.forecast !== null)
    .sort((left, right) => left.date.localeCompare(right.date))
  const orderedRecords = [...actualRecords, ...forecastRecords.filter((record) => record.actual === null)]
  return (
    <div>
      <div className="table-scroll">
        <table className="demand-table">
          <caption>
            Showing {orderedRecords.length} rows: {actualRecords.length} actual sales and {forecastRecords.length} forecasts. All quantities are units.
            <span>All recorded sales are shown newest first, followed by the complete forecast horizon.</span>
            <span>Future actual sales are not yet recorded. Historical forecast values and bounds are not available in this forecast dataset.</span>
          </caption>
          <colgroup><col className="date-column" /><col className="period-column" /><col /><col /><col /><col /></colgroup>
          <thead><tr><th scope="col">Date</th><th scope="col">Period</th><th scope="col" className="numeric">Actual sales</th><th scope="col" className="numeric">Forecast demand</th><th scope="col" className="numeric">Lower bound</th><th scope="col" className="numeric">Upper bound</th></tr></thead>
          <tbody>{orderedRecords.map((record, index) => (
            <tr key={`${record.date}-${index}`}>
              <td>{formatDate(record.date)}</td>
              <td>{record.forecast === null ? 'Historical' : 'Forecast'}</td>
              <td className="numeric">{record.actual === null ? 'Not yet recorded' : formatNumber(record.actual, 1)}</td>
              <td className="numeric">{record.forecast === null ? 'Not available' : formatNumber(record.forecast, 1)}</td>
              <td className="numeric">{record.confidence?.[0] == null ? 'Not available' : formatNumber(record.confidence[0], 1)}</td>
              <td className="numeric">{record.confidence?.[1] == null ? 'Not available' : formatNumber(record.confidence[1], 1)}</td>
            </tr>
          ))}{!orderedRecords.length && <tr><td colSpan={6}>No sales or forecast rows are available for this selection.</td></tr>}</tbody>
        </table>
      </div>
    </div>
  )
}

function SalesTable({ records, loading }: { records: SalesRecord[]; loading: boolean }) {
  if (loading) return <div className="loading-state"><span className="spinner" />Loading sales history…</div>
  if (!records.length) return <div className="table-empty">No historical sales rows are available.</div>
  return (
    <div className="table-scroll">
      <table>
        <thead><tr><th>Date</th><th>Store</th><th>Item</th><th className="numeric">Units sold</th></tr></thead>
        <tbody>{records.map((record, index) => <tr key={`${record.date}-${index}`}><td>{formatDate(record.date)}</td><td>{storeName(record.store)}</td><td>{itemName(record.item)}</td><td className="numeric">{formatNumber(record.sales)}</td></tr>)}</tbody>
      </table>
    </div>
  )
}

function InventoryResults({ inventory }: { inventory: InventoryResponse }) {
  const fields = [
    ['Lead-time demand', 'Lead Time Demand'],
    ['Safety stock', 'Safety Stock'],
    ['Reorder point', 'Reorder Point'],
    ['Suggested order', 'Suggested Reorder Qty'],
  ] as const
  const recommendation = inventory.recommendation
  const isLow = Boolean(recommendation['Low-Stock Alert'])
  return (
    <>
      <div className={`stock-alert ${isLow ? 'stock-low' : 'stock-ok'}`}><span>{isLow ? 'Review replenishment' : 'Stock appears sufficient'}</span><strong>{isLow ? 'Reorder suggested' : 'No immediate reorder'}</strong></div>
      <div className="recommendation-metrics">{fields.map(([label, key]) => <div key={key}><small>{label}</small><strong>{formatNumber(Number(recommendation[key]))}<span> units</span></strong></div>)}</div>
      <p className="method-note">{inventory.note}</p>
    </>
  )
}

export default App

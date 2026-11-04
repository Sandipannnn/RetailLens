import { useEffect, useState } from 'react'
import type { ChartPoint } from './ForecastChart'

interface ComparisonRow {
  date: string
  actual_sales: number
  yhat: number
  yhat_lower: number
  yhat_upper: number
}

interface Props {
  records: ChartPoint[]
  loading: boolean
  comparisonUrl: string
  formatDate: (date: string) => string
  formatNumber: (value: number, digits?: number) => string
}

export default function SalesForecastTable({ records, loading, comparisonUrl, formatDate, formatNumber }: Props) {
  const [view, setView] = useState<'comparison' | 'future' | 'history'>('comparison')
  const [comparison, setComparison] = useState<ChartPoint[]>([])
  const [pending, setPending] = useState(true)
  const [error, setError] = useState('')
  useEffect(() => {
    const controller = new AbortController()
    fetch(comparisonUrl, { signal: controller.signal })
      .then(async (response) => {
        const result = await response.json()
        if (!response.ok) throw new Error(result.detail ?? 'Unable to load historical comparison.')
        return result.records as ComparisonRow[]
      })
      .then((rows) => {
        if (controller.signal.aborted) return
        setComparison(rows.map((row) => ({ date: row.date, actual: row.actual_sales,
          forecast: row.yhat, confidence: [row.yhat_lower, row.yhat_upper] })))
      })
      .catch((cause: unknown) => {
        if (!controller.signal.aborted) setError(cause instanceof Error ? cause.message : 'Unable to load comparison.')
      })
      .finally(() => { if (!controller.signal.aborted) setPending(false) })
    return () => controller.abort()
  }, [comparisonUrl])

  const rows = (view === 'comparison' ? comparison : records.filter((row) =>
    view === 'future' ? row.forecast !== null : row.actual !== null))
    .slice().sort((a, b) => view === 'future' ? a.date.localeCompare(b.date) : b.date.localeCompare(a.date))
  const showActual = view !== 'future'
  const showForecast = view !== 'history'
  const headings = ['Date', ...(showActual ? ['Actual sales'] : []),
    ...(showForecast ? ['Forecast demand', 'Lower bound', 'Upper bound'] : [])]
  const values = (row: ChartPoint) => [row.date, ...(showActual ? [row.actual] : []),
    ...(showForecast ? [row.forecast, row.confidence?.[0], row.confidence?.[1]] : [])]
  const busy = loading || (view === 'comparison' && pending)

  function download() {
    const csv = [headings, ...rows.map(values)].map((row) => row.map((value) =>
      `"${String(value ?? '').replaceAll('"', '""')}"`).join(',')).join('\n')
    const url = URL.createObjectURL(new Blob([csv], { type: 'text/csv;charset=utf-8' }))
    const link = document.createElement('a')
    link.href = url
    link.download = `retail-lens-${view}.csv`
    link.click()
    window.setTimeout(() => URL.revokeObjectURL(url), 0)
  }

  return <div>
    <div className="demand-table-filter">
      <label>Show <select value={view} onChange={(event) => setView(event.target.value as typeof view)}>
        <option value="comparison">Actual vs forecast — XGBoost historical comparison</option>
        <option value="future">Future forecasts</option>
        <option value="history">All recorded sales</option>
      </select></label>
      <button type="button" className="secondary-button" onClick={download} disabled={busy || !rows.length}>Download this table</button>
    </div>
    {busy ? <div className="loading-state"><span className="spinner" />Loading {view === 'comparison' ? 'historical predictions and bounds' : 'data'}…</div>
      : view === 'comparison' && error ? <p className="form-error" role="alert">{error}</p>
        : <div className="table-scroll" key={view}><table className="demand-table">
          <caption>
            {rows.length} rows. All quantities are units.
            <span>{view === 'comparison'
              ? 'XGBoost historical comparison: actual sales and one-day-ahead predictions for the last 90 recorded days. Bounds use errors from the preceding 90 days. This comparison uses XGBoost regardless of the future model selector.'
              : view === 'future' ? 'Future predictions from the selected forecast model. Actual sales exist only in the recorded sales period.'
                : 'Complete recorded sales for the selected date range. Use the historical comparison to compare actual sales with predictions.'}</span>
          </caption>
          <thead><tr>{headings.map((heading, index) => <th scope="col" key={heading} className={index ? 'numeric' : ''}>{heading}</th>)}</tr></thead>
          <tbody>{rows.map((row) => <tr key={row.date}>{values(row).map((value, index) =>
            <td key={index} className={index ? 'numeric' : ''}>{index === 0 ? formatDate(String(value)) : value == null ? '—' : formatNumber(Number(value), 1)}</td>)}</tr>)}
            {!rows.length && <tr><td colSpan={headings.length}>No rows in this date range. Choose a range including the latest recorded sales to see historical comparisons.</td></tr>}
          </tbody>
        </table></div>}
  </div>
}

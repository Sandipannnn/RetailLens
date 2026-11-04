/** Prefer model predictions; otherwise summarize actual recent daily sales. */
export function demandMetrics(
  sales: { date: string; sales: number }[],
  predictions: number[],
  horizon: number,
) {
  const isForecast = predictions.length > 0
  const actual = [...sales].sort((a, b) => a.date.localeCompare(b.date)).slice(-horizon)
  const values = isForecast ? predictions : actual.map((row) => row.sales)
  const total = values.length ? values.reduce((sum, value) => sum + value, 0) : null
  return { isForecast, total, average: total === null ? null : total / values.length, days: values.length }
}

export type ChartGranularity = 'month' | 'year'

export interface AggregatableChartPoint {
  date: string
  actual: number | null
  forecast: number | null
  confidence?: [number, number]
}

function parseDate(value: string): Date | null {
  const date = new Date(`${value.slice(0, 10)}T00:00:00Z`)
  return Number.isNaN(date.getTime()) ? null : date
}

/** Use monthly buckets for ranges up to one calendar year and yearly buckets otherwise. */
export function chartGranularity(start: string, end: string): ChartGranularity {
  const startDate = parseDate(start)
  const endDate = parseDate(end)
  if (!startDate || !endDate || endDate < startDate) return 'month'

  const oneYearLater = new Date(startDate)
  oneYearLater.setUTCFullYear(oneYearLater.getUTCFullYear() + 1)
  return endDate <= oneYearLater ? 'month' : 'year'
}

export function aggregateChartData(
  points: AggregatableChartPoint[],
  granularity: ChartGranularity,
): AggregatableChartPoint[] {
  const buckets = new Map<string, AggregatableChartPoint>()

  points.forEach((point) => {
    const date = parseDate(point.date)
    if (!date) return
    const year = date.getUTCFullYear()
    const month = granularity === 'month' ? date.getUTCMonth() : 0
    const key = `${year}-${String(month + 1).padStart(2, '0')}`
    const bucket = buckets.get(key)
    if (!bucket) {
      const bucket: AggregatableChartPoint = {
        date: `${key}-01`,
        actual: point.actual,
        forecast: point.forecast,
      }
      if (point.confidence) bucket.confidence = [...point.confidence] as [number, number]
      buckets.set(key, bucket)
      return
    }

    if (point.actual !== null) bucket.actual = (bucket.actual ?? 0) + point.actual
    if (point.forecast !== null) bucket.forecast = (bucket.forecast ?? 0) + point.forecast
    if (point.confidence) {
      bucket.confidence = [
        (bucket.confidence?.[0] ?? 0) + point.confidence[0],
        (bucket.confidence?.[1] ?? 0) + point.confidence[1],
      ]
    }
  })

  return [...buckets.values()].sort((left, right) => left.date.localeCompare(right.date))
}

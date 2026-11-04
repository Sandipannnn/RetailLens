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

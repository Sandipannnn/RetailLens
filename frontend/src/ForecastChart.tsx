export interface ChartPoint {
  date: string
  actual: number | null
  forecast: number | null
  confidence?: [number, number]
}

interface ForecastChartProps {
  data: ChartPoint[]
  formatDate: (value: string) => string
  compactDate: (value: string) => string
  formatNumber: (value: number, digits?: number) => string
}

const WIDTH = 1000
const HEIGHT = 340
const LEFT = 58
const RIGHT = 18
const TOP = 18
const BOTTOM = 42

function linePath(
  data: ChartPoint[],
  series: 'actual' | 'forecast',
  xAt: (index: number) => number,
  yAt: (value: number) => number,
): string {
  const points = data.flatMap((point, index) => {
    const value = point[series]
    return value === null ? [] : [{ index, value }]
  })
  return points
    .map(({ index, value }, indexInSeries) => {
      const command = indexInSeries === 0 ? 'M' : 'L'
      return `${command}${xAt(index).toFixed(2)},${yAt(value).toFixed(2)}`
    })
    .join(' ')
}

export default function ForecastChart({
  data,
  formatDate,
  compactDate,
  formatNumber,
}: ForecastChartProps) {
  if (!data.length) return null

  const allValues = data.flatMap((point) => [
    point.actual,
    point.forecast,
    point.confidence?.[0] ?? null,
    point.confidence?.[1] ?? null,
  ]).filter((value): value is number => value !== null && Number.isFinite(value))
  const minimum = Math.min(...allValues)
  const maximum = Math.max(...allValues)
  const padding = Math.max((maximum - minimum) * 0.1, maximum * 0.025, 1)
  const domainMin = minimum - padding
  const domainMax = maximum + padding
  const plotWidth = WIDTH - LEFT - RIGHT
  const plotHeight = HEIGHT - TOP - BOTTOM
  const xAt = (index: number) => LEFT + (index / Math.max(data.length - 1, 1)) * plotWidth
  const yAt = (value: number) => TOP + ((domainMax - value) / (domainMax - domainMin)) * plotHeight
  const actualPath = linePath(data, 'actual', xAt, yAt)
  const forecastPath = linePath(data, 'forecast', xAt, yAt)
  const confidencePoints = data.flatMap((point, index) => {
    const range = point.confidence
    return range ? [{ index, lower: range[0], upper: range[1] }] : []
  })
  const confidencePath = confidencePoints.length > 1
    ? [
      ...confidencePoints.map(({ index, upper }) => `${xAt(index).toFixed(2)},${yAt(upper).toFixed(2)}`),
      ...[...confidencePoints].reverse().map(({ index, lower }) => `${xAt(index).toFixed(2)},${yAt(lower).toFixed(2)}`),
    ].join(' ')
    : ''
  const yTicks = Array.from({ length: 5 }, (_, index) => domainMax - (index / 4) * (domainMax - domainMin))
  const xTickIndexes = Array.from(new Set([0, Math.round((data.length - 1) / 4), Math.round((data.length - 1) / 2), Math.round(3 * (data.length - 1) / 4), data.length - 1]))

  return (
    <svg
      className="forecast-svg"
      viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
      role="img"
      aria-label="Historical sales and forecast with uncertainty interval"
      preserveAspectRatio="none"
    >
      {yTicks.map((tick, index) => {
        const y = yAt(tick)
        return (
          <g key={`y-${index}`}>
            <line x1={LEFT} x2={WIDTH - RIGHT} y1={y} y2={y} className="chart-grid-line" />
            <text x={LEFT - 11} y={y + 3} textAnchor="end" className="chart-axis-label">{formatNumber(tick)}</text>
          </g>
        )
      })}
      {confidencePath && <polygon points={confidencePath} className="confidence-band" />}
      {actualPath && <path d={actualPath} className="actual-line" />}
      {forecastPath && <path d={forecastPath} className="forecast-line" />}
      {data.map((point, index) => point.actual === null ? null : (
        <circle key={`actual-${point.date}`} cx={xAt(index)} cy={yAt(point.actual)} r="2.1" className="actual-point">
          <title>{`${formatDate(point.date)} · Actual ${formatNumber(point.actual, 1)}`}</title>
        </circle>
      ))}
      {data.map((point, index) => point.forecast === null ? null : (
        <circle key={`forecast-${point.date}`} cx={xAt(index)} cy={yAt(point.forecast)} r="2.3" className="forecast-point">
          <title>{`${formatDate(point.date)} · Forecast ${formatNumber(point.forecast, 1)}`}</title>
        </circle>
      ))}
      {xTickIndexes.map((index) => (
        <text key={`x-${index}`} x={xAt(index)} y={HEIGHT - 12} textAnchor="middle" className="chart-axis-label">
          {compactDate(data[index].date)}
        </text>
      ))}
    </svg>
  )
}

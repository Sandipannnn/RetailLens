import { test } from 'node:test'
import assert from 'node:assert/strict'
import { demandMetrics } from './demandMetrics.ts'

test('missing forecasts show actual recent sales, not an invented prediction', () => {
  const sales = [{date: '2017-12-31', sales: 30}, {date: '2017-12-29', sales: 100}, {date: '2017-12-30', sales: 10}]
  assert.deepEqual(demandMetrics(sales, [], 2), {isForecast: false, total: 40, average: 20, days: 2})
})

test('new predictions replace historical values, including valid zero forecasts', () => {
  assert.deepEqual(demandMetrics([{date: '2017-12-31', sales: 30}], [0, 0], 2),
    {isForecast: true, total: 0, average: 0, days: 2})
})

test('an empty date range stays empty rather than fabricating sales', () => {
  assert.deepEqual(demandMetrics([], [], 30), {isForecast: false, total: null, average: null, days: 0})
})

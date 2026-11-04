import { test } from 'node:test'
import assert from 'node:assert/strict'
import { aggregateChartData, chartGranularity, demandMetrics } from './demandMetrics.ts'

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

test('uses monthly chart buckets for a range up to one year', () => {
  assert.equal(chartGranularity('2017-01-01', '2017-12-31'), 'month')
  assert.deepEqual(aggregateChartData([
    {date: '2017-01-02', actual: 10, forecast: null},
    {date: '2017-01-20', actual: 5, forecast: null},
    {date: '2017-02-01', actual: 8, forecast: null},
  ], 'month'), [
    {date: '2017-01-01', actual: 15, forecast: null},
    {date: '2017-02-01', actual: 8, forecast: null},
  ])
})

test('uses yearly chart buckets for a range longer than one year', () => {
  assert.equal(chartGranularity('2013-01-01', '2017-12-31'), 'year')
  assert.deepEqual(aggregateChartData([
    {date: '2013-01-02', actual: 10, forecast: null},
    {date: '2013-12-20', actual: 5, forecast: null},
    {date: '2014-01-01', actual: 8, forecast: null},
  ], 'year'), [
    {date: '2013-01-01', actual: 15, forecast: null},
    {date: '2014-01-01', actual: 8, forecast: null},
  ])
})

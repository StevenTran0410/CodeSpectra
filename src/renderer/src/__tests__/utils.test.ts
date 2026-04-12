import { describe, it, expect } from 'vitest'
import { normConf } from '../lib/reportUtils'
import { toErrorMessage } from '../lib/errors'

describe('normConf', () => {
  it('returns "high" when input is "high"', () => {
    expect(normConf('high')).toBe('high')
  })

  it('returns "medium" when input is "medium"', () => {
    expect(normConf('medium')).toBe('medium')
  })

  it('returns "low" when input is "low"', () => {
    expect(normConf('low')).toBe('low')
  })

  it('returns "medium" for undefined', () => {
    expect(normConf(undefined)).toBe('medium')
  })

  it('returns "medium" for an unrecognised string', () => {
    expect(normConf('unknown')).toBe('medium')
  })

  it('returns "medium" for an empty string', () => {
    expect(normConf('')).toBe('medium')
  })
})

describe('toErrorMessage', () => {
  it('extracts the message from an Error instance', () => {
    const err = new Error('something went wrong')
    expect(toErrorMessage(err)).toBe('something went wrong')
  })

  it('converts a plain string to itself', () => {
    expect(toErrorMessage('network timeout')).toBe('network timeout')
  })

  it('converts a number via String()', () => {
    expect(toErrorMessage(404)).toBe('404')
  })

  it('converts null to "null"', () => {
    expect(toErrorMessage(null)).toBe('null')
  })
})

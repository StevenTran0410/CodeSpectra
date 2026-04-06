import log from 'electron-log'

// File transport: cap at 5 MB, keep 3 rotated files (15 MB total)
log.transports.file.level = 'info'
log.transports.file.maxSize = 5 * 1024 * 1024
log.transports.file.archiveLog = (oldLogFile) => {
  const { dir, name, ext } = require('path').parse(oldLogFile.toString())
  const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19)
  return require('path').join(dir, `${name}.${timestamp}${ext}`)
}

log.transports.console.level = 'debug'

// Scrub credential patterns from all log messages before writing
const _REDACT = [
  { re: /(Bearer\s+)[A-Za-z0-9\-._~+/]+=*/gi, replace: '$1[REDACTED]' },
  { re: /("?api[_-]?key"?\s*[:=]\s*["']?)[A-Za-z0-9\-_]{8,}(["']?)/gi, replace: '$1[REDACTED]$2' },
  { re: /(sk-[A-Za-z0-9]{8})[A-Za-z0-9\-_]+/g, replace: '$1[REDACTED]' },
]

log.hooks.push((message) => {
  message.data = message.data.map((item) => {
    if (typeof item !== 'string') return item
    let s = item
    for (const { re, replace } of _REDACT) s = s.replace(re, replace)
    return s
  })
  return message
})

export const logger = log

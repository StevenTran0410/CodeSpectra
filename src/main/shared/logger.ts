import log from 'electron-log'

log.transports.file.level = 'info'
log.transports.console.level = 'debug'

export const logger = log

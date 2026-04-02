import log from 'electron-log'
import { app } from 'electron'
import path from 'path'

log.transports.file.resolvePathFn = () =>
  path.join(app.getPath('userData'), 'logs', 'main.log')

log.transports.file.level = 'info'
log.transports.console.level = process.env.NODE_ENV === 'development' ? 'debug' : 'warn'

export const logger = log.scope('main')
export default log

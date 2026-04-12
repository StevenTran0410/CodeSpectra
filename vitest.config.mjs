import { defineConfig } from 'vitest/config'

export default defineConfig({
  test: {
    // Use 'node' for pure utility tests. Switch individual test files to jsdom
    // via the @vitest-environment docblock when testing DOM/React components.
    environment: 'node',
    globals: true,
  },
})

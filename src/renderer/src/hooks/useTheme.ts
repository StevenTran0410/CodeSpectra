import { useState, useEffect } from 'react'

function applyTheme(theme: 'dark' | 'light'): void {
  if (theme === 'dark') {
    document.documentElement.classList.add('dark')
  } else {
    document.documentElement.classList.remove('dark')
  }
}

export function useTheme(): { theme: 'dark' | 'light'; isDark: boolean; toggle: () => void } {
  const [theme, setTheme] = useState<'dark' | 'light'>(() => {
    const saved = (localStorage.getItem('theme') as 'dark' | 'light') ?? 'dark'
    // Apply immediately to avoid flash on init
    applyTheme(saved)
    return saved
  })

  useEffect(() => {
    applyTheme(theme)
    localStorage.setItem('theme', theme)
  }, [theme])

  const toggle = (): void => setTheme((t) => (t === 'dark' ? 'light' : 'dark'))

  return { theme, isDark: theme === 'dark', toggle }
}

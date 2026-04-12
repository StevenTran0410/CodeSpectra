import React, { useState } from 'react'
import { Eye, EyeOff, Search } from 'lucide-react'

interface InputProps extends React.InputHTMLAttributes<HTMLInputElement> {
  variant?: 'text' | 'password' | 'search'
  error?: string
  className?: string
}

export const Input = React.forwardRef<HTMLInputElement, InputProps>(
  ({ variant = 'text', error, className = '', ...props }, ref) => {
    const [showPassword, setShowPassword] = useState(false)

    const baseClass = 'input'
    const errorClass = error ? 'border-red-500 focus:ring-red-500' : ''
    const finalClass = `${baseClass} ${errorClass} ${className}`.trim()

    if (variant === 'password') {
      return (
        <div className="relative">
          <input
            ref={ref}
            type={showPassword ? 'text' : 'password'}
            className={finalClass}
            {...props}
          />
          <button
            type="button"
            onClick={() => setShowPassword(!showPassword)}
            className="absolute right-2 top-1/2 -translate-y-1/2 text-gray-500 hover:text-gray-400 transition-colors"
            aria-label={showPassword ? 'Hide password' : 'Show password'}
          >
            {showPassword ? <EyeOff size={18} /> : <Eye size={18} />}
          </button>
          {error && <p className="text-xs text-red-400 mt-1">{error}</p>}
        </div>
      )
    }

    if (variant === 'search') {
      return (
        <div className="relative">
          <Search className="absolute left-2 top-1/2 -translate-y-1/2 text-gray-500" size={18} />
          <input
            ref={ref}
            type="text"
            className={`${finalClass} pl-8`}
            {...props}
          />
          {error && <p className="text-xs text-red-400 mt-1">{error}</p>}
        </div>
      )
    }

    return (
      <div>
        <input
          ref={ref}
          type="text"
          className={finalClass}
          {...props}
        />
        {error && <p className="text-xs text-red-400 mt-1">{error}</p>}
      </div>
    )
  }
)

Input.displayName = 'Input'

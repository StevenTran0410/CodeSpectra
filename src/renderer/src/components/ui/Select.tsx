import React from 'react'
import { ChevronDown } from 'lucide-react'
import { Spinner } from './Spinner'

interface SelectProps extends React.SelectHTMLAttributes<HTMLSelectElement> {
  loading?: boolean
  error?: string
  placeholder?: string
  className?: string
  children: React.ReactNode
}

export const Select = React.forwardRef<HTMLSelectElement, SelectProps>(
  ({ loading = false, error, placeholder, className = '', children, ...props }, ref) => {
    const baseClass = 'input appearance-none pr-8'
    const errorClass = error ? 'border-red-500 focus:ring-red-500' : ''
    const finalClass = `${baseClass} ${errorClass} ${className}`.trim()

    return (
      <div>
        <div className="relative">
          <select
            ref={ref}
            className={finalClass}
            {...props}
          >
            {placeholder && (
              <option value="" disabled>
                {placeholder}
              </option>
            )}
            {children}
          </select>
          {loading ? (
            <div className="absolute right-2 top-1/2 -translate-y-1/2 text-gray-500 pointer-events-none">
              <Spinner size="sm" />
            </div>
          ) : (
            <ChevronDown
              size={18}
              className="absolute right-2 top-1/2 -translate-y-1/2 text-gray-500 pointer-events-none"
            />
          )}
        </div>
        {error && <p className="text-xs text-red-400 mt-1">{error}</p>}
      </div>
    )
  }
)

Select.displayName = 'Select'

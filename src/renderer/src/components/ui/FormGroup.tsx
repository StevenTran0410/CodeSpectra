import React from 'react'

interface FormGroupProps {
  label?: string
  helperText?: string
  error?: string
  required?: boolean
  className?: string
  children: React.ReactNode
}

export const FormGroup: React.FC<FormGroupProps> = ({
  label,
  helperText,
  error,
  required = false,
  className = '',
  children
}) => {
  return (
    <div className={`space-y-1.5 ${className}`.trim()}>
      {label && (
        <label className="block text-xs text-gray-400">
          {label}
          {required && <span className="text-red-400 ml-1">*</span>}
        </label>
      )}
      {children}
      {helperText && !error && <p className="text-xs text-gray-500">{helperText}</p>}
      {error && <p className="text-xs text-red-400">{error}</p>}
    </div>
  )
}

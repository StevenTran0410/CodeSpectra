import React from 'react'
import { Spinner } from './Spinner'

type ButtonVariant = 'primary' | 'secondary' | 'danger' | 'ghost' | 'link'
type ButtonSize = 'sm' | 'md' | 'lg'

interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: ButtonVariant
  size?: ButtonSize
  loading?: boolean
  leftIcon?: React.ReactNode
  rightIcon?: React.ReactNode
}

const variantClasses: Record<ButtonVariant, string> = {
  primary: 'btn-primary',
  secondary: 'btn-secondary',
  danger: 'btn-danger',
  ghost: 'btn-ghost',
  link: 'text-sm text-blue-400 hover:text-blue-300 underline-offset-2 hover:underline'
}

const sizeClasses: Record<ButtonSize, string> = {
  sm: 'text-xs px-2.5 py-1',
  md: '',
  lg: 'text-base px-5 py-2.5'
}

export const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  (
    {
      variant = 'primary',
      size = 'md',
      loading = false,
      leftIcon,
      rightIcon,
      disabled,
      children,
      className,
      ...props
    },
    ref
  ) => {
    const baseClass = variantClasses[variant]
    const sizeClass = sizeClasses[size]
    const isDisabled = disabled || loading
    const finalClass = `${baseClass} ${sizeClass} ${className ?? ''}`.trim()

    return (
      <button
        ref={ref}
        type="button"
        disabled={isDisabled}
        className={finalClass}
        {...props}
      >
        <div className="inline-flex items-center gap-1.5">
          {loading && <Spinner size="sm" />}
          {!loading && leftIcon && <span>{leftIcon}</span>}
          {children}
          {rightIcon && <span>{rightIcon}</span>}
        </div>
      </button>
    )
  }
)

Button.displayName = 'Button'

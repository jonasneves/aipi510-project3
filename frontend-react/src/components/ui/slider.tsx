import * as React from "react"
import { cn } from "../../lib/utils"

interface SliderProps extends Omit<React.InputHTMLAttributes<HTMLInputElement>, 'type'> {
  label?: string
}

const Slider = React.forwardRef<HTMLInputElement, SliderProps>(
  ({ className, label, ...props }, ref) => {
    return (
      <div className="space-y-2">
        {label && (
          <div className="flex justify-between">
            <span className="text-sm text-muted-foreground">{label}</span>
            <span className="text-sm font-medium">{props.value}</span>
          </div>
        )}
        <input
          type="range"
          className={cn(
            "w-full h-2 bg-secondary rounded-lg appearance-none cursor-pointer accent-primary",
            className
          )}
          ref={ref}
          {...props}
        />
      </div>
    )
  }
)
Slider.displayName = "Slider"

export { Slider }

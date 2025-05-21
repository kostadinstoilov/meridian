// Define the basic structure for your logs
interface LogEntry {
  level: 'debug' | 'info' | 'warn' | 'error';
  message: string;
  timestamp: string;
  context?: Record<string, unknown>;
  error?: {
    message: string;
    stack?: string;
    cause?: unknown;
  };
}

// Basic logger class
export class Logger {
  private baseContext: Record<string, unknown>;

  constructor(baseContext: Record<string, unknown> = {}) {
    // Clone the context to prevent mutation issues if the source object changes
    this.baseContext = { ...baseContext };
  }

  // Method to create a "child" logger with additional context
  child(additionalContext: Record<string, unknown>): Logger {
    return new Logger({ ...this.baseContext, ...additionalContext });
  }

  // Central logging function
  private log(level: LogEntry['level'], message: string, context?: Record<string, unknown>, error?: Error) {
    const entry: LogEntry = {
      level,
      message,
      timestamp: new Date().toISOString(),
      // Merge base context, method-specific context
      context: { ...this.baseContext, ...context },
    };

    if (error) {
      entry.error = {
        message: error.message,
        stack: error.stack,
        // Include cause if available
        ...(error.cause ? { cause: error.cause } : {}),
      };
    }

    // The core idea: output structured JSON via console.log
    // Logpush / Tail Workers will pick this up.
    console.log(JSON.stringify(entry));
  }

  // Convenience methods for different levels
  debug(message: string, context?: Record<string, unknown>) {
    this.log('debug', message, context);
  }

  info(message: string, context?: Record<string, unknown>) {
    this.log('info', message, context);
  }

  warn(message: string, context?: Record<string, unknown>, error?: Error) {
    this.log('warn', message, context, error);
  }

  error(message: string, context?: Record<string, unknown>, error?: Error) {
    this.log('error', message, context, error);
  }
}

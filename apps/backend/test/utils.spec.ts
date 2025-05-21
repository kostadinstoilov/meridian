import type { Context } from 'hono';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import type { HonoEnv } from '../src/app';
import { hasValidAuthToken } from '../src/lib/utils';

describe('hasValidAuthToken', () => {
  // Mock Context object
  let mockContext: Context<HonoEnv>;
  const validToken = 'valid-token-12345';

  beforeEach(() => {
    // Reset mocks
    vi.resetAllMocks();

    // Create a mock context with request headers and environment
    mockContext = {
      req: {
        header: vi.fn(),
      },
      env: {
        API_TOKEN: validToken,
      },
    } as unknown as Context<HonoEnv>;
  });

  it('should return true when Authorization header has the correct Bearer token', () => {
    // Setup header mock to return the valid token
    mockContext.req.header = vi.fn().mockImplementation((name: string) => {
      if (name === 'Authorization') return `Bearer ${validToken}`;
      return undefined;
    });

    // Call the function
    const result = hasValidAuthToken(mockContext);

    // Assert
    expect(result).toBe(true);
    expect(mockContext.req.header).toHaveBeenCalledWith('Authorization');
  });

  it('should return false when Authorization header is missing', () => {
    // Setup header mock to return undefined
    mockContext.req.header = vi.fn().mockImplementation((name: string) => {
      return undefined;
    });

    // Call the function
    const result = hasValidAuthToken(mockContext);

    // Assert
    expect(result).toBe(false);
    expect(mockContext.req.header).toHaveBeenCalledWith('Authorization');
  });

  it('should return false when Authorization header has incorrect token value', () => {
    // Setup header mock to return an invalid token
    mockContext.req.header = vi.fn().mockImplementation((name: string) => {
      if (name === 'Authorization') return 'Bearer wrong-token';
      return undefined;
    });

    // Call the function
    const result = hasValidAuthToken(mockContext);

    // Assert
    expect(result).toBe(false);
    expect(mockContext.req.header).toHaveBeenCalledWith('Authorization');
  });

  it('should return false when Authorization header uses a scheme other than Bearer', () => {
    // Setup header mock to return a non-Bearer token
    mockContext.req.header = vi.fn().mockImplementation((name: string) => {
      if (name === 'Authorization') return `Basic ${validToken}`;
      return undefined;
    });

    // Call the function
    const result = hasValidAuthToken(mockContext);

    // Assert
    expect(result).toBe(false);
    expect(mockContext.req.header).toHaveBeenCalledWith('Authorization');
  });

  it('should return false when API_TOKEN environment variable is not set or empty', () => {
    // Mock the environment with an empty API_TOKEN
    mockContext.env.API_TOKEN = '';

    // Setup header mock to return a valid token format
    mockContext.req.header = vi.fn().mockImplementation((name: string) => {
      if (name === 'Authorization') return `Bearer ${validToken}`;
      return undefined;
    });

    // Call the function
    const result = hasValidAuthToken(mockContext);

    // Assert
    expect(result).toBe(false);
    expect(mockContext.req.header).toHaveBeenCalledWith('Authorization');
  });
});

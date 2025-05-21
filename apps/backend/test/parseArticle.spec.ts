import { Readability } from '@mozilla/readability';
import * as linkedom from 'linkedom';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { parseArticle } from '../src/lib/parsers';

// Mock the Readability and parseHTML dependencies
vi.mock('@mozilla/readability', () => {
  return {
    Readability: vi.fn(),
  };
});

vi.mock('linkedom', () => {
  return {
    parseHTML: vi.fn(),
  };
});

describe('parseArticle', () => {
  // Note: Testing Readability itself is hard. Focus on the wrapper.

  beforeEach(() => {
    vi.resetAllMocks();

    // Default mocks for linkedom
    vi.mocked(linkedom.parseHTML).mockReturnValue({
      document: 'mock-document',
    } as unknown);
  });

  it('should return an error Result if Readability constructor or parse() throws an exception', () => {
    // Setup: Make Readability throw an error
    vi.mocked(Readability).mockImplementation(() => {
      throw new Error('Readability error');
    });

    // Execute
    const result = parseArticle({ html: '<html><body>Test</body></html>' });

    // Verify
    expect(result.isErr()).toBe(true);
    if (result.isErr()) {
      expect(result.error.type).toBe('READABILITY_ERROR');
    }
  });

  it('should return an error Result if Readability returns null', () => {
    // Setup: Make Readability.parse() return null
    vi.mocked(Readability).mockImplementation(() => {
      return {
        parse: () => null,
      } as unknown as Readability;
    });

    // Execute
    const result = parseArticle({ html: '<html><body>Test</body></html>' });

    // Verify
    expect(result.isErr()).toBe(true);
    if (result.isErr()) {
      expect(result.error.type).toBe('NO_ARTICLE_FOUND');
    }
  });

  it('should return an error Result if Readability result is missing title', () => {
    // Setup: Make Readability.parse() return an object without a title
    vi.mocked(Readability).mockImplementation(() => {
      return {
        parse: () => ({
          title: '', // empty title
          textContent: 'Some content',
        }),
      } as unknown as Readability;
    });

    // Execute
    const result = parseArticle({ html: '<html><body>Test</body></html>' });

    // Verify
    expect(result.isErr()).toBe(true);
    if (result.isErr()) {
      expect(result.error.type).toBe('NO_ARTICLE_FOUND');
    }
  });

  it('should return an error Result if Readability result is missing textContent', () => {
    // Setup: Make Readability.parse() return an object without textContent
    vi.mocked(Readability).mockImplementation(() => {
      return {
        parse: () => ({
          title: 'Article Title',
          textContent: '', // empty textContent
        }),
      } as unknown as Readability;
    });

    // Execute
    const result = parseArticle({ html: '<html><body>Test</body></html>' });

    // Verify
    expect(result.isErr()).toBe(true);
    if (result.isErr()) {
      expect(result.error.type).toBe('NO_ARTICLE_FOUND');
    }
  });

  it('should return the extracted title, cleaned textContent, and publishedTime when successful', () => {
    // Setup: Make Readability.parse() return a valid article
    vi.mocked(Readability).mockImplementation(() => {
      return {
        parse: () => ({
          title: 'Article Title',
          textContent: 'Article content here',
          publishedTime: '2025-03-18T18:04:44-04:00',
        }),
      } as unknown as Readability;
    });

    // Execute
    const result = parseArticle({ html: '<html><body>Test</body></html>' });

    // Verify
    expect(result.isOk()).toBe(true);
    if (result.isOk()) {
      expect(result.value).toEqual({
        title: 'Article Title',
        text: 'Article content here',
        publishedTime: '2025-03-18T18:04:44-04:00',
      });
    }
  });

  it('should clean and normalize whitespace in the extracted textContent', () => {
    // Setup: Make Readability.parse() return messy text content
    const messyText = '  Multiple    spaces  \n\n\n  and \t\t tabs \n   and extra newlines  ';
    vi.mocked(Readability).mockImplementation(() => {
      return {
        parse: () => ({
          title: 'Article Title',
          textContent: messyText,
        }),
      } as unknown as Readability;
    });

    // Execute
    const result = parseArticle({ html: '<html><body>Test</body></html>' });

    // Verify
    expect(result.isOk()).toBe(true);
    if (result.isOk()) {
      // The text should be cleaned according to the cleanString function logic
      expect(result.value.text).toBe('Multiple spaces\nand tabs\nand extra newlines');
    }
  });
});
